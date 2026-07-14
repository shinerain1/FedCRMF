import math
from collections import OrderedDict

import numpy as np
import torch

from ..server import FedAvg


def compute_fedcrmf_gated_current_responses(
    history,
    alpha,
    mu,
    active=True,
    risk_mode="centered",
):
    """
    Solve the FedCRMF coordinate aggregation problem.

    Args:
        history: Client response history shaped [L, M, ...].
        alpha: Reference client weights shaped [M].
        mu: Coordinate risk regularization strength.
        active: If false, return the unmodified current responses.

    Returns:
        effective_current: omega * r^(t), shaped [M, ...].
        omega: Coordinate gate 1 / (1 + mu * s).
        risk_score: alpha-weighted risk score / sqrt(L).
        raw_domain_norm: alpha-weighted centered or raw response norm.
        shared_strength: Alpha-weighted RMS shared-response magnitude.
    """
    if history.dim() < 2:
        raise ValueError("history must have shape [L, M, ...]")
    if history.shape[0] < 1 or history.shape[1] < 1:
        raise ValueError("history must contain at least one round and one client")
    if alpha.numel() != history.shape[1]:
        raise ValueError("alpha length must match the number of clients")
    if float(mu) < 0.0:
        raise ValueError("mu must be non-negative")

    hist = history.float()
    alpha = alpha.to(device=hist.device, dtype=hist.dtype)
    alpha = alpha / alpha.sum().clamp_min(torch.finfo(hist.dtype).eps)
    alpha_view = alpha.view(1, -1, *([1] * (hist.dim() - 2)))

    shared = (alpha_view * hist).sum(dim=1, keepdim=True)
    domain = hist - shared

    history_length = hist.shape[0]
    risk_mode = str(risk_mode).strip().lower()
    if risk_mode in {"centered", "domain", "pd"}:
        risk_response = domain
    elif risk_mode in {"raw", "no_centering", "nocentering"}:
        risk_response = hist
    else:
        raise ValueError(f"Unsupported FedCRMF risk_mode: {risk_mode}")
    raw_domain_norm = (
        alpha_view * risk_response.pow(2)
    ).sum(dim=(0, 1)).sqrt()
    risk_score = raw_domain_norm / math.sqrt(float(history_length))
    shared_strength = (
        (alpha_view * shared.expand_as(hist).pow(2)).sum(dim=(0, 1))
        / float(history_length)
    ).sqrt()

    if active:
        omega = 1.0 / (1.0 + float(mu) * risk_score)
    else:
        omega = torch.ones_like(risk_score)

    effective_current = omega.unsqueeze(0) * hist[-1]
    return (
        effective_current,
        omega,
        risk_score,
        raw_domain_norm,
        shared_strength,
    )


class FedCRMFServer(FedAvg):
    """
    Coordinate-risk regularized full-update aggregation.

    For coordinate k, the server estimates

        s_k = ||P_d R_k||_{F,alpha} / sqrt(L)

    and solves

        min_delta 0.5 * (delta - u_k)^2 + 0.5 * mu * s_k * delta^2,

    where u_k is the current FedAvg update. The closed-form update is

        delta_k = omega_k * u_k,
        omega_k = 1 / (1 + mu * s_k).
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.fedcrmf_history_length = max(
            int(hparam.get("fedcrmf_history_length", 2)),
            1,
        )
        self.fedcrmf_warmup_rounds = max(
            int(
                hparam.get(
                    "fedcrmf_warmup_rounds",
                    self.fedcrmf_history_length - 1,
                )
            ),
            0,
        )
        self.fedcrmf_mu = max(float(hparam.get("fedcrmf_mu", 10000.0)), 0.0)
        self.fedcrmf_gate_variant = str(
            hparam.get("fedcrmf_gate_variant", "full")
        ).strip().lower()
        valid_variants = {
            "full",
            "uniform_shrinkage",
            "permuted_gate",
            "no_centering",
            "response_gate_dropout",
        }
        if self.fedcrmf_gate_variant not in valid_variants:
            raise ValueError(
                f"Unsupported fedcrmf_gate_variant={self.fedcrmf_gate_variant!r}. "
                f"Valid: {sorted(valid_variants)}"
            )
        self.fedcrmf_risk_mode = "raw" if self.fedcrmf_gate_variant == "no_centering" else "centered"
        self.fedcrmf_gate_dropout_p = min(
            max(float(hparam.get("fedcrmf_gate_dropout_p", 0.5)), 0.0),
            1.0,
        )
        self.fedcrmf_alpha_mode = str(
            hparam.get("fedcrmf_alpha_mode", "uniform")
        ).strip().lower()
        if self.fedcrmf_alpha_mode not in {"uniform", "q"}:
            self.fedcrmf_alpha_mode = "uniform"
        self.fedcrmf_eps = max(
            float(hparam.get("fedcrmf_eps", 1e-8)),
            1e-12,
        )

        raw_hist_keys = str(hparam.get("fedcrmf_hist_keys", "ALL")).strip()
        if raw_hist_keys.lower() in {"", "all", "*", "none"}:
            self.fedcrmf_key_patterns = []
        else:
            self.fedcrmf_key_patterns = [
                item.strip()
                for item in raw_hist_keys.split(",")
                if item.strip()
            ]
        self.fedcrmf_hist_max_numel = int(
            hparam.get("fedcrmf_hist_max_numel", 5_000_000)
        )

        self._fedcrmf_response_history = {}
        self._fedcrmf_history_client_indices = None
        self._fedcrmf_last_omega_by_key = {}

        self.hparam["fedcrmf_history_length"] = self.fedcrmf_history_length
        self.hparam["fedcrmf_warmup_rounds"] = self.fedcrmf_warmup_rounds
        self.hparam["fedcrmf_mu"] = self.fedcrmf_mu
        self.hparam["fedcrmf_gate_variant"] = self.fedcrmf_gate_variant
        self.hparam["fedcrmf_risk_mode"] = self.fedcrmf_risk_mode
        self.hparam["fedcrmf_gate_dropout_p"] = self.fedcrmf_gate_dropout_p
        self.hparam["fedcrmf_alpha_mode"] = self.fedcrmf_alpha_mode
        self.hparam["fedcrmf_hist_keys"] = (
            ",".join(self.fedcrmf_key_patterns)
            if self.fedcrmf_key_patterns
            else "ALL"
        )
        self.hparam["fedcrmf_hist_max_numel"] = self.fedcrmf_hist_max_numel

    def _use_key(self, key, param):
        if not torch.is_floating_point(param):
            return False
        if param.numel() > self.fedcrmf_hist_max_numel:
            return False
        if not self.fedcrmf_key_patterns:
            return True
        key_lower = key.lower()
        return any(
            pattern.lower() in key_lower
            for pattern in self.fedcrmf_key_patterns
        )

    def _alpha_weights(self, coefficients, num_clients):
        if self.fedcrmf_alpha_mode == "q":
            alpha = torch.tensor(coefficients, dtype=torch.float32)
            return alpha / alpha.sum().clamp_min(self.fedcrmf_eps)
        return torch.full(
            (num_clients,),
            1.0 / float(max(num_clients, 1)),
            dtype=torch.float32,
        )

    def _update_history(self, key, deltas):
        history = self._fedcrmf_response_history.setdefault(key, [])
        history.append(deltas.detach().cpu().float())
        if len(history) > self.fedcrmf_history_length:
            del history[: len(history) - self.fedcrmf_history_length]
        return history

    def _build_effective_responses(
        self,
        last_weights,
        local_states,
        sampled_client_indices,
        coefficients,
    ):
        current_indices = tuple(sampled_client_indices)
        if (
            self._fedcrmf_history_client_indices is not None
            and self._fedcrmf_history_client_indices != current_indices
        ):
            self._fedcrmf_response_history.clear()
        self._fedcrmf_history_client_indices = current_indices

        alpha = self._alpha_weights(
            coefficients,
            len(sampled_client_indices),
        )
        q = torch.tensor(coefficients, dtype=torch.float32)
        q = q / q.sum().clamp_min(self.fedcrmf_eps)
        control_ready = self._round >= self.fedcrmf_warmup_rounds

        effective_responses = OrderedDict()
        self._fedcrmf_last_omega_by_key = {}
        total = controlled = 0
        omega_means = []
        omega_mins = []
        gate_dropout_keep_means = []
        risk_score_means = []
        raw_domain_norm_means = []
        shared_strength_means = []

        for key, global_param in last_weights.items():
            if not self._use_key(key, global_param):
                continue

            stack = torch.stack(
                [state[key] for state in local_states],
                dim=0,
            ).float()
            deltas = stack - global_param.float().unsqueeze(0)
            history = self._update_history(key, deltas)
            hist = torch.stack(history, dim=0)
            (
                effective_current,
                omega,
                risk_score,
                raw_domain_norm,
                shared_strength,
            ) = compute_fedcrmf_gated_current_responses(
                hist,
                alpha,
                self.fedcrmf_mu,
                active=control_ready,
                risk_mode=self.fedcrmf_risk_mode,
            )
            if control_ready and self.fedcrmf_gate_variant == "uniform_shrinkage":
                current = hist[-1]
                q_view = q.view(-1, *([1] * (current.dim() - 1)))
                target_update = (q_view * effective_current).sum(dim=0)
                raw_update = (q_view * current).sum(dim=0)
                shrink = (
                    target_update.norm()
                    / raw_update.norm().clamp_min(self.fedcrmf_eps)
                ).clamp(0.0, 1.0)
                omega = torch.full_like(omega, float(shrink.item()))
                effective_current = omega.unsqueeze(0) * current
            elif control_ready and self.fedcrmf_gate_variant == "permuted_gate" and omega.numel() > 1:
                stable_key_seed = sum(
                    (index + 1) * byte
                    for index, byte in enumerate(key.encode("utf-8"))
                )
                shift = (
                    stable_key_seed
                    + 1000003 * int(self._round)
                    + 9176 * int(self.hparam.get("seed", 0))
                ) % omega.numel()
                flat_omega = omega.reshape(-1)
                omega = torch.roll(flat_omega, shifts=int(shift), dims=0).view_as(omega)
                effective_current = omega.unsqueeze(0) * hist[-1]
            elif control_ready and self.fedcrmf_gate_variant == "response_gate_dropout":
                keep_prob = 1.0 - self.fedcrmf_gate_dropout_p
                if omega.numel() > 1 and keep_prob < 1.0:
                    generator = torch.Generator(device="cpu")
                    stable_key_seed = sum(
                        (index + 1) * byte
                        for index, byte in enumerate(key.encode("utf-8"))
                    )
                    seed = (
                        stable_key_seed
                        + 1000003 * int(self._round)
                        + 9176 * int(self.hparam.get("seed", 0))
                    ) % (2**31)
                    generator.manual_seed(seed)
                    z = (
                        torch.rand(
                            omega.shape,
                            generator=generator,
                            dtype=omega.dtype,
                        )
                        < keep_prob
                    ).to(dtype=omega.dtype)
                    layer_mean = omega.mean()
                    omega = layer_mean + z * (omega - layer_mean)
                    gate_dropout_keep_means.append(float(z.mean().item()))
                else:
                    gate_dropout_keep_means.append(1.0)
                effective_current = omega.unsqueeze(0) * hist[-1]
            effective_responses[key] = effective_current.cpu()
            self._fedcrmf_last_omega_by_key[key] = omega.detach().cpu()

            total += global_param.numel()
            controlled += int((omega < 1.0 - 1e-7).sum().item())
            omega_means.append(float(omega.mean().item()))
            omega_mins.append(float(omega.min().item()))
            risk_score_means.append(float(risk_score.mean().item()))
            raw_domain_norm_means.append(
                float(raw_domain_norm.mean().item())
            )
            shared_strength_means.append(
                float(shared_strength.mean().item())
            )

        self._last_round_extra_metrics.update(
            {
                "fedcrmf_controlled_param_ratio": (
                    controlled / total if total else 0.0
                ),
                "fedcrmf_mean_omega": (
                    float(np.mean(omega_means)) if omega_means else 1.0
                ),
                "fedcrmf_min_omega": (
                    float(np.mean(omega_mins)) if omega_mins else 1.0
                ),
                "fedcrmf_gate_dropout_keep_ratio": (
                    float(np.mean(gate_dropout_keep_means))
                    if gate_dropout_keep_means
                    else "N/A"
                ),
                "fedcrmf_mean_risk_score": (
                    float(np.mean(risk_score_means))
                    if risk_score_means
                    else 0.0
                ),
                "fedcrmf_mean_raw_domain_norm": (
                    float(np.mean(raw_domain_norm_means))
                    if raw_domain_norm_means
                    else 0.0
                ),
                "fedcrmf_mean_shared_strength": (
                    float(np.mean(shared_strength_means))
                    if shared_strength_means
                    else 0.0
                ),
                "fedcrmf_q_alpha_l1": float(
                    torch.abs(q - alpha).sum().item()
                ),
                "fedcrmf_history_size": min(
                    self._round + 1,
                    self.fedcrmf_history_length,
                ),
            }
        )
        return effective_responses

    def aggregate(self, sampled_client_indices, coefficients):
        last_weights = OrderedDict(
            (key, value.detach().clone().cpu())
            for key, value in self.model.state_dict().items()
        )
        local_states = [
            OrderedDict(
                (key, value.detach().clone().cpu())
                for key, value
                in self.clients[index].model.state_dict().items()
            )
            for index in sampled_client_indices
        ]
        self._update_pairwise_client_update_metrics(
            last_weights,
            local_states,
        )
        effective_responses = self._build_effective_responses(
            last_weights,
            local_states,
            sampled_client_indices,
            coefficients,
        )

        q = torch.tensor(coefficients, dtype=torch.float32)
        q = q / q.sum().clamp_min(self.fedcrmf_eps)
        aggregated = OrderedDict()
        for key, global_param in last_weights.items():
            if torch.is_floating_point(global_param):
                if key in effective_responses:
                    responses = effective_responses[key].float()
                    q_view = q.view(
                        -1,
                        *([1] * (responses.dim() - 1)),
                    )
                    update = (q_view * responses).sum(dim=0)
                    aggregated[key] = (
                        global_param.float() + update
                    ).to(dtype=global_param.dtype)
                else:
                    stack = torch.stack(
                        [state[key] for state in local_states],
                        dim=0,
                    ).float()
                    q_view = q.view(
                        -1,
                        *([1] * (stack.dim() - 1)),
                    )
                    aggregated[key] = (q_view * stack).sum(dim=0).to(
                        dtype=global_param.dtype
                    )
            else:
                aggregated[key] = local_states[0][key]
        self.model.load_state_dict(aggregated)
