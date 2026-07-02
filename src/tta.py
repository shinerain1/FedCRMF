import copy
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _softmax_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _split_tta_mode(mode):
    if mode == "tent":
        return "entropy", "tent", None, False, False
    if mode == "fedcrmf_gated_tent":
        return "entropy", "tent", None, True, False
    if mode == "full_tent":
        return "entropy", "tent", "all", False, False
    if mode == "fedcrmf_gated_full_tent":
        return "entropy", "tent", "all", True, False
    if mode == "tent_frozen_bn":
        return "entropy", "frozen", None, False, False
    if mode == "lr_enhanced_tent_frozen_bn":
        return "entropy", "frozen", None, False, True
    if mode == "fedcrmf_gated_tent_frozen_bn":
        return "entropy", "frozen", None, True, False
    if mode == "pl_full_tta":
        return "pseudo_label", "frozen", "all", False, False
    if mode == "lr_enhanced_pl_full_tta":
        return "pseudo_label", "frozen", "all", False, True
    if mode == "fedcrmf_gated_pl_full_tta":
        return "pseudo_label", "frozen", "all", True, False
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _configure_bn_modules(model, bn_mode):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if bn_mode == "tent":
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            elif bn_mode == "frozen":
                module.eval()
            else:
                raise ValueError(f"Unsupported TTA BN mode: {bn_mode}")


def _configure_tta_model(model, param_scope, bn_mode):
    if bn_mode == "tent":
        model.train()
    elif bn_mode == "frozen":
        model.eval()
    else:
        raise ValueError(f"Unsupported TTA BN mode: {bn_mode}")
    _configure_bn_modules(model, bn_mode)
    model.requires_grad_(False)

    selected = OrderedDict()
    if param_scope == "all":
        for name, param in model.named_parameters():
            param.requires_grad_(True)
            selected[name] = param
        return selected

    if param_scope != "bn_affine":
        raise ValueError(f"Unsupported TTA parameter scope: {param_scope}")

    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(True)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                param.requires_grad_(True)
                selected[full_name] = param
    if not selected:
        raise RuntimeError("Tent found no BatchNorm affine parameters.")
    return selected


def _make_gate_map(
    selected_params,
    omega_by_key,
    device,
    gate_mode,
    rho,
    gate_transform="square_norm",
    gate_power=2.0,
    gate_norm_scope="global",
    gate_clip_min=None,
    gate_clip_max=None,
    eps=1e-12,
):
    gate_mode = str(gate_mode).strip().lower()
    if gate_mode not in {"suppress", "enhance"}:
        raise ValueError(f"Unsupported TTA gate mode: {gate_mode}")
    gate_transform = str(gate_transform).strip().lower()
    if gate_transform not in {"linear", "square_norm"}:
        raise ValueError(f"Unsupported TTA gate transform: {gate_transform}")
    gate_norm_scope = str(gate_norm_scope).strip().lower()
    if gate_norm_scope not in {"global", "layer"}:
        raise ValueError(f"Unsupported TTA gate norm scope: {gate_norm_scope}")
    gate_power = max(float(gate_power), 1.0)
    rho = min(max(float(rho), 0.0), 1.0)
    gate_clip_min = (
        None
        if gate_clip_min in (None, "", "none", "None")
        else float(gate_clip_min)
    )
    gate_clip_max = (
        None
        if gate_clip_max in (None, "", "none", "None")
        else float(gate_clip_max)
    )
    if (
        gate_clip_min is not None
        and gate_clip_max is not None
        and gate_clip_min > gate_clip_max
    ):
        raise ValueError("tta_gate_clip_min cannot be larger than tta_gate_clip_max")
    gate_map = {}
    missing = []
    flat_values = []
    raw_gate_values = []
    transformed_gate_values = []
    raw_gate_map = OrderedDict()
    normalization_map = {}
    for name, param in selected_params.items():
        omega = None
        if omega_by_key is not None:
            omega = omega_by_key.get(name)
        if omega is None:
            missing.append(name)
            raw_gate = torch.zeros_like(param, device=device)
        else:
            raw_gate = 1.0 - omega.to(device=device, dtype=param.dtype)
            raw_gate = raw_gate.view_as(param).clamp(0.0, 1.0)
        raw_gate_map[name] = raw_gate
        raw_gate_values.append(raw_gate.detach().reshape(-1).float().cpu())
        if gate_transform == "square_norm" and gate_norm_scope == "layer":
            layer_power_mean = raw_gate.detach().float().pow(gate_power).mean()
            normalization_map[name] = 1.0 / (layer_power_mean + float(eps))

    if raw_gate_values:
        raw_flat = torch.cat(raw_gate_values)
        raw_mean = raw_flat.mean()
        raw_square_mean = raw_flat.pow(2).mean()
        raw_power_mean = raw_flat.pow(gate_power).mean()
        if gate_transform == "square_norm" and gate_norm_scope == "global":
            normalization = 1.0 / (raw_power_mean + float(eps))
        else:
            normalization = torch.tensor(1.0, dtype=raw_flat.dtype)
    else:
        raw_flat = None
        raw_mean = None
        raw_square_mean = None
        raw_power_mean = None
        normalization = None

    for name, raw_gate in raw_gate_map.items():
        if gate_transform == "square_norm":
            current_norm = (
                normalization_map[name]
                if gate_norm_scope == "layer"
                else normalization
            )
            gate_strength = raw_gate.pow(gate_power) * current_norm.to(
                device=raw_gate.device, dtype=raw_gate.dtype
            )
        else:
            gate_strength = raw_gate
        transformed_gate_values.append(
            gate_strength.detach().reshape(-1).float().cpu()
        )
        gate_map[name] = gate_strength

    if transformed_gate_values:
        transformed_flat = torch.cat(transformed_gate_values)
        gate_strength_mean = transformed_flat.mean()
    else:
        transformed_flat = None
        gate_strength_mean = None

    for name, gate_strength in gate_map.items():
        if gate_mode == "enhance":
            centered_strength = gate_strength - gate_strength_mean.to(
                device=gate_strength.device, dtype=gate_strength.dtype
            )
            gate = 1.0 + rho * centered_strength
        else:
            gate = gate_strength
        if gate_clip_min is not None or gate_clip_max is not None:
            min_value = -float("inf") if gate_clip_min is None else gate_clip_min
            max_value = float("inf") if gate_clip_max is None else gate_clip_max
            gate = gate.clamp(min=min_value, max=max_value)
        flat_values.append(gate.detach().reshape(-1).float().cpu())
        gate_map[name] = gate
    if flat_values:
        flat = torch.cat(flat_values)
        stats = {
            "gate_mode": gate_mode,
            "gate_transform": gate_transform,
            "gate_norm_scope": gate_norm_scope,
            "gate_power": gate_power,
            "gate_rho": rho,
            "gate_clip_min": "N/A" if gate_clip_min is None else gate_clip_min,
            "gate_clip_max": "N/A" if gate_clip_max is None else gate_clip_max,
            "gate_mean": float(flat.mean().item()),
            "gate_min": float(flat.min().item()),
            "gate_max": float(flat.max().item()),
            "gate_numel": int(flat.numel()),
            "raw_gate_mean": float(raw_flat.mean().item()),
            "raw_gate_min": float(raw_flat.min().item()),
            "raw_gate_max": float(raw_flat.max().item()),
            "raw_gate_square_mean": float(raw_square_mean.item()),
            "raw_gate_power_mean": float(raw_power_mean.item()),
            "gate_strength_mean": float(transformed_flat.mean().item()),
            "gate_strength_min": float(transformed_flat.min().item()),
            "gate_strength_max": float(transformed_flat.max().item()),
            "gate_strength_norm": float(normalization.item()),
        }
    else:
        stats = {
            "gate_mode": gate_mode,
            "gate_transform": gate_transform,
            "gate_norm_scope": gate_norm_scope,
            "gate_power": gate_power,
            "gate_rho": rho,
            "gate_clip_min": "N/A" if gate_clip_min is None else gate_clip_min,
            "gate_clip_max": "N/A" if gate_clip_max is None else gate_clip_max,
            "gate_mean": "N/A",
            "gate_min": "N/A",
            "gate_max": "N/A",
            "gate_numel": 0,
            "raw_gate_mean": "N/A",
            "raw_gate_min": "N/A",
            "raw_gate_max": "N/A",
            "raw_gate_square_mean": "N/A",
            "raw_gate_power_mean": "N/A",
            "gate_strength_mean": "N/A",
            "gate_strength_min": "N/A",
            "gate_strength_max": "N/A",
            "gate_strength_norm": "N/A",
        }
    stats["missing_gate_params"] = len(missing)
    stats["missing_gate_param_names"] = missing[:20]
    return gate_map, stats


def _clone_state(selected_params):
    return {
        name: param.detach().clone()
        for name, param in selected_params.items()
    }


def _l2_anchor_loss(selected_params, initial_state, beta):
    if beta <= 0.0:
        return None
    loss = None
    for name, param in selected_params.items():
        anchor = initial_state[name].to(device=param.device, dtype=param.dtype)
        value = (param - anchor).pow(2).sum()
        loss = value if loss is None else loss + value
    return 0.5 * float(beta) * loss


def _evaluate_metric(dataset, y_pred, y_true, metadata):
    metric, _ = dataset.eval(
        y_pred.to("cpu"),
        y_true.to("cpu"),
        metadata.to("cpu"),
    )
    return metric


def _evaluate_source_model(model, dataloader, ds_bundle, device, max_batches):
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    _configure_bn_modules(model, "frozen")
    model.requires_grad_(False)

    entropies = []
    confidences = []
    preds = []
    labels_all = []
    metadata_all = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            data, labels, metadata = batch[0], batch[1], batch[2]
            if isinstance(metadata, list):
                metadata = metadata[0]
            data = data.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device)

            logits = model(data)
            entropy = _softmax_entropy(logits)
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values

            entropies.append(entropy.detach().cpu())
            confidences.append(confidence.detach().cpu())
            preds.append(torch.argmax(logits.detach(), dim=-1).cpu())
            labels_all.append(labels.detach().cpu())
            metadata_all.append(metadata.detach().cpu())

    y_true = torch.cat(labels_all) if labels_all else torch.empty(0)
    metadata = torch.cat(metadata_all) if metadata_all else torch.empty(0)
    y_pred = torch.cat(preds) if preds else torch.empty(0)
    metric = _evaluate_metric(ds_bundle.dataset, y_pred, y_true, metadata)
    entropy = torch.cat(entropies) if entropies else torch.empty(0)
    confidence = torch.cat(confidences) if confidences else torch.empty(0)
    model.to("cpu")
    return {
        "metric": metric,
        "entropy": float(entropy.mean().item()) if entropy.numel() else "N/A",
        "confidence": (
            float(confidence.mean().item()) if confidence.numel() else "N/A"
        ),
        "num_samples": int(y_true.numel()),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _run_one_tta_mode(
    base_model,
    dataloader,
    ds_bundle,
    device,
    mode,
    omega_by_key,
    lr,
    beta,
    param_scope,
    optimizer_name,
    confidence_threshold,
    gate_mode,
    gate_transform,
    gate_power,
    gate_norm_scope,
    gate_clip_min,
    gate_clip_max,
    rho,
    max_batches,
    reset_each_batch,
):
    model = copy.deepcopy(base_model)
    model.to(device)
    (
        objective_name,
        bn_mode,
        scope_override,
        use_gate,
        use_global_lr_enhance,
    ) = _split_tta_mode(mode)
    effective_param_scope = scope_override or param_scope
    selected_params = _configure_tta_model(model, effective_param_scope, bn_mode)
    initial_state = _clone_state(selected_params)
    source_model = None
    if objective_name == "pseudo_label":
        source_model = copy.deepcopy(base_model)
        source_model.to(device)
        source_model.eval()
        _configure_bn_modules(source_model, "frozen")
        source_model.requires_grad_(False)
    rho = min(max(float(rho), 0.0), 1.0)
    gate_map = None
    gate_stats = {}
    if use_gate or use_global_lr_enhance:
        computed_gate_map, gate_stats = _make_gate_map(
            selected_params,
            omega_by_key,
            device,
            gate_mode,
            rho,
            gate_transform,
            gate_power,
            gate_norm_scope,
            gate_clip_min,
            gate_clip_max,
        )
        if use_gate:
            gate_map = computed_gate_map

    if use_global_lr_enhance:
        gate_mean = gate_stats.get("gate_mean", 1.0)
        lr_multiplier = float(gate_mean) if gate_mean != "N/A" else 1.0
    else:
        lr_multiplier = 1.0
    effective_lr = float(lr) * lr_multiplier
    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(selected_params.values(), lr=effective_lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(selected_params.values(), lr=effective_lr)
    else:
        raise ValueError(f"Unsupported TTA optimizer: {optimizer_name}")

    source_eval = _evaluate_source_model(
        base_model,
        dataloader,
        ds_bundle,
        device,
        max_batches,
    )

    before_entropies = []
    after_entropies = []
    before_confidences = []
    after_confidences = []
    before_preds = []
    after_preds = []
    labels_all = []
    metadata_all = []
    batch_rows = []
    selected_counts = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if reset_each_batch:
            with torch.no_grad():
                for name, param in selected_params.items():
                    param.copy_(initial_state[name].to(param.device))

        data, labels, metadata = batch[0], batch[1], batch[2]
        if isinstance(metadata, list):
            metadata = metadata[0]
        data = data.to(device)
        labels = labels.to(device)
        metadata = metadata.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits_before = model(data)
        entropy_before = _softmax_entropy(logits_before)
        confidence_before = F.softmax(logits_before, dim=-1).max(dim=-1).values
        selected_count = labels.new_tensor(0)
        if objective_name == "entropy":
            loss = entropy_before.mean()
            selected_count = labels.new_tensor(labels.numel())
        elif objective_name == "pseudo_label":
            with torch.no_grad():
                source_logits = source_model(data)
                source_probs = F.softmax(source_logits, dim=-1)
                source_confidence, pseudo_labels = source_probs.max(dim=-1)
                selected_mask = source_confidence >= float(confidence_threshold)
            selected_count = selected_mask.sum()
            if int(selected_count.item()) > 0:
                loss = F.cross_entropy(
                    logits_before[selected_mask],
                    pseudo_labels[selected_mask],
                )
            else:
                loss = None
        else:
            raise ValueError(f"Unsupported TTA objective: {objective_name}")
        anchor_loss = _l2_anchor_loss(selected_params, initial_state, beta)
        if loss is not None and anchor_loss is not None:
            loss = loss + anchor_loss
        if loss is not None:
            loss.backward()
            if gate_map is not None:
                for name, param in selected_params.items():
                    if param.grad is not None:
                        param.grad.mul_(gate_map[name])
            optimizer.step()

        with torch.no_grad():
            logits_after = model(data)
            entropy_after = _softmax_entropy(logits_after)
            confidence_after = F.softmax(logits_after, dim=-1).max(dim=-1).values

        before_entropies.append(entropy_before.detach().cpu())
        after_entropies.append(entropy_after.detach().cpu())
        before_confidences.append(confidence_before.detach().cpu())
        after_confidences.append(confidence_after.detach().cpu())
        before_preds.append(torch.argmax(logits_before.detach(), dim=-1).cpu())
        after_preds.append(torch.argmax(logits_after.detach(), dim=-1).cpu())
        labels_all.append(labels.detach().cpu())
        metadata_all.append(metadata.detach().cpu())
        selected_counts.append(int(selected_count.detach().cpu().item()))
        batch_rows.append(
            {
                "mode": mode,
                "batch": batch_idx,
                "objective": objective_name,
                "selected_samples": int(selected_count.detach().cpu().item()),
                "selected_ratio": float(
                    selected_count.detach().cpu().item() / max(labels.numel(), 1)
                ),
                "entropy_before": float(entropy_before.mean().item()),
                "entropy_after": float(entropy_after.mean().item()),
                "entropy_delta": float(
                    entropy_after.mean().item() - entropy_before.mean().item()
                ),
                "confidence_before": float(confidence_before.mean().item()),
                "confidence_after": float(confidence_after.mean().item()),
            }
        )

    y_true = torch.cat(labels_all) if labels_all else torch.empty(0)
    metadata = torch.cat(metadata_all) if metadata_all else torch.empty(0)
    y_before = torch.cat(before_preds) if before_preds else torch.empty(0)
    y_after = torch.cat(after_preds) if after_preds else torch.empty(0)
    metric_online_before = _evaluate_metric(
        ds_bundle.dataset,
        y_before,
        y_true,
        metadata,
    )
    metric_after = _evaluate_metric(ds_bundle.dataset, y_after, y_true, metadata)
    entropy_before = torch.cat(before_entropies)
    entropy_after = torch.cat(after_entropies)
    confidence_before = torch.cat(before_confidences)
    confidence_after = torch.cat(after_confidences)

    summary = {
        "mode": mode,
        "objective": objective_name,
        "use_gate": bool(use_gate),
        "bn_mode": bn_mode,
        "param_scope": effective_param_scope,
        "optimizer": optimizer_name,
        "confidence_threshold": float(confidence_threshold),
        "base_lr": float(lr),
        "lr": float(effective_lr),
        "lr_multiplier": float(lr_multiplier),
        "use_global_lr_enhance": bool(use_global_lr_enhance),
        "tta_rho": float(rho),
        "beta": float(beta),
        "reset_each_batch": bool(reset_each_batch),
        "num_batches": len(batch_rows),
        "num_samples": int(y_true.numel()),
        "selected_samples": int(sum(selected_counts)),
        "selected_ratio": float(
            sum(selected_counts) / max(int(y_true.numel()), 1)
        ),
        "source_entropy": source_eval["entropy"],
        "source_confidence": source_eval["confidence"],
        "entropy_before": source_eval["entropy"],
        "online_entropy_before": float(entropy_before.mean().item()),
        "entropy_after": float(entropy_after.mean().item()),
        "entropy_delta": float(
            entropy_after.mean().item() - float(source_eval["entropy"])
            if source_eval["entropy"] != "N/A"
            else "nan"
        ),
        "confidence_before": source_eval["confidence"],
        "online_confidence_before": float(confidence_before.mean().item()),
        "confidence_after": float(confidence_after.mean().item()),
        "confidence_delta": float(
            confidence_after.mean().item() - float(source_eval["confidence"])
            if source_eval["confidence"] != "N/A"
            else "nan"
        ),
        "metric_before": _jsonable(source_eval["metric"]),
        "metric_online_before": _jsonable(metric_online_before),
        "metric_after": _jsonable(metric_after),
        **gate_stats,
    }
    model.to("cpu")
    if source_model is not None:
        source_model.to("cpu")
    return summary, batch_rows


def run_tta_comparison(server, output_dir):
    hparam = server.hparam
    if not _as_bool(hparam.get("tta_eval", False)):
        return None
    split_name = str(hparam.get("tta_split", "test"))
    if split_name not in server.test_dataloader:
        raise RuntimeError(f"TTA split {split_name!r} is not available.")

    lr = float(hparam.get("tta_lr", 1e-4))
    beta = float(hparam.get("tta_beta", 0.0))
    param_scope = str(hparam.get("tta_param_scope", "bn_affine")).strip()
    optimizer_name = str(hparam.get("tta_optimizer", "sgd")).strip()
    confidence_threshold = float(hparam.get("tta_conf_threshold", 0.9))
    gate_mode = str(hparam.get("tta_gate_mode", "enhance")).strip().lower()
    gate_transform = str(
        hparam.get("tta_gate_transform", "square_norm")
    ).strip().lower()
    gate_power = float(hparam.get("tta_gate_power", 2.0))
    gate_norm_scope = str(hparam.get("tta_gate_norm_scope", "global")).strip().lower()
    gate_clip_min = hparam.get("tta_gate_clip_min", None)
    gate_clip_max = hparam.get("tta_gate_clip_max", None)
    rho = float(hparam.get("tta_rho", 1.0))
    max_batches = int(hparam.get("tta_max_batches", 0))
    reset_each_batch = _as_bool(hparam.get("tta_reset_each_batch", False))
    modes = [
        item.strip()
        for item in str(
            hparam.get("tta_modes", "pl_full_tta,fedcrmf_gated_pl_full_tta")
        ).split(",")
        if item.strip()
    ]
    omega_by_key = getattr(server, "_fedcrmf_last_omega_by_key", None) or getattr(server, "_core_last_omega_by_key", None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    batch_rows = []
    for mode in modes:
        summary, mode_batch_rows = _run_one_tta_mode(
            server.model,
            server.test_dataloader[split_name],
            server.ds_bundle,
            server.device,
            mode,
            omega_by_key,
            lr,
            beta,
            param_scope,
            optimizer_name,
            confidence_threshold,
            gate_mode,
            gate_transform,
            gate_power,
            gate_norm_scope,
            gate_clip_min,
            gate_clip_max,
            rho,
            max_batches,
            reset_each_batch,
        )
        rows.append(summary)
        batch_rows.extend(mode_batch_rows)

    with (output_dir / "tta_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    summary_fields = [
        "mode",
        "objective",
        "use_gate",
        "bn_mode",
        "param_scope",
        "optimizer",
        "confidence_threshold",
        "base_lr",
        "lr",
        "lr_multiplier",
        "use_global_lr_enhance",
        "tta_rho",
        "beta",
        "reset_each_batch",
        "num_batches",
        "num_samples",
        "selected_samples",
        "selected_ratio",
        "source_entropy",
        "source_confidence",
        "entropy_before",
        "online_entropy_before",
        "entropy_after",
        "entropy_delta",
        "confidence_before",
        "online_confidence_before",
        "confidence_after",
        "confidence_delta",
        "gate_mean",
        "gate_min",
        "gate_max",
        "gate_numel",
        "gate_mode",
        "gate_transform",
        "gate_norm_scope",
        "gate_power",
        "gate_rho",
        "gate_clip_min",
        "gate_clip_max",
        "raw_gate_mean",
        "raw_gate_min",
        "raw_gate_max",
        "raw_gate_square_mean",
        "raw_gate_power_mean",
        "gate_strength_mean",
        "gate_strength_min",
        "gate_strength_max",
        "gate_strength_norm",
        "missing_gate_params",
    ]
    metric_key = server.ds_bundle.key_metric
    with (output_dir / "tta_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=summary_fields
            + [
                f"{metric_key}_before",
                f"{metric_key}_online_before",
                f"{metric_key}_after",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in summary_fields}
            out[f"{metric_key}_before"] = row["metric_before"].get(metric_key, "")
            out[f"{metric_key}_online_before"] = row[
                "metric_online_before"
            ].get(metric_key, "")
            out[f"{metric_key}_after"] = row["metric_after"].get(metric_key, "")
            writer.writerow(out)

    with (output_dir / "tta_per_batch.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "mode",
                "batch",
                "objective",
                "selected_samples",
                "selected_ratio",
                "entropy_before",
                "entropy_after",
                "entropy_delta",
                "confidence_before",
                "confidence_after",
            ],
        )
        writer.writeheader()
        writer.writerows(batch_rows)
    return rows
