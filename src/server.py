import copy
import csv
import json
import time
import os
from pathlib import Path

from multiprocessing import pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *

import wandb

class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam['num_rounds']
        self.fraction = hparam['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None
        self._last_round_extra_metrics = {}
        self._last_round_layer_metrics = {}

    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))

    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices


    def update_clients(self, sampled_client_indices):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(self._round)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        for idx in tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()


    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        last_weights = OrderedDict((k, v.detach().clone().cpu()) for k, v in self.model.state_dict().items())
        local_states = [
            OrderedDict((k, v.detach().clone().cpu()) for k, v in self.clients[idx].model.state_dict().items())
            for idx in sampled_client_indices
        ]
        self._update_pairwise_client_update_metrics(last_weights, local_states)
        averaged_weights = OrderedDict()
        q = torch.tensor(coefficients, dtype=torch.float32)
        q = q / q.sum().clamp_min(torch.finfo(q.dtype).eps)
        for key, global_value in last_weights.items():
            if torch.is_floating_point(global_value):
                stacked = torch.stack(
                    [state[key].to(torch.float32) for state in local_states],
                    dim=0,
                )
                q_view = q.view(-1, *([1] * (stacked.dim() - 1)))
                averaged_weights[key] = (q_view * stacked).sum(dim=0).to(
                    dtype=global_value.dtype
                )
            else:
                # Integer buffers such as BatchNorm counters cannot be
                # meaningfully averaged. They are identical in the current
                # frozen-BN setup, so preserve one client's value.
                averaged_weights[key] = local_states[0][key]
        self.model.load_state_dict(averaged_weights)


    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)

    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
                # print("DEBUG: server.py:183")
                # break
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            print(metric)
            if getattr(self.device, "type", str(self.device)) == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0
        last_metric_dict = {}
        metrics_per_round = []

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_federated_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader)
                metric_dict[name] = metric

                if name == 'val':
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val

            print(metric_dict)
            last_metric_dict = metric_dict
            round_extra = {}
            if isinstance(getattr(self, "_last_round_extra_metrics", None), dict):
                round_extra = copy.deepcopy(self._last_round_extra_metrics)
            metrics_per_round.append((r, metric_dict, round_extra))
            if self.hparam['wandb']:
                wandb.log(metric_dict, step=self._round*self.hparam['local_epochs'])
            self._round += 1

        # Model checkpoint saving disabled by user request; record only.
        if self.hparam['wandb']:
            if "id_val" in self.test_dataloader:
                wandb.summary['best_id_round'] = best_id_val_round
                wandb.summary['best_id_val_acc'] = best_id_val_value
                wandb.summary['best_id_selected_id_test_acc'] = best_id_val_test_value
            if "val" in self.test_dataloader:
                wandb.summary['best_lodo_round'] = best_lodo_val_round
                wandb.summary['best_lodo_val_acc'] = best_lodo_val_value
                wandb.summary['best_lodo_selected_test_acc'] = best_lodo_val_test_value
        else:
            print(f"best_id_round: {best_id_val_round}")
            print(f"best_id_val_acc: {best_id_val_value}")
            print(f"best_id_selected_id_test_acc: {best_id_val_test_value}")
            print(f"best_lodo_round: {best_lodo_val_round}")
            print(f"best_lodo_val_acc: {best_lodo_val_value}")
            print(f"best_lodo_selected_test_acc: {best_lodo_val_test_value}")
        self.save_record(
            last_metric_dict,
            metrics_per_round,
            best_id_val_round,
            best_id_val_value,
            best_id_val_test_value,
            best_lodo_val_round,
            best_lodo_val_value,
            best_lodo_val_test_value,
        )
        save_single_model = self.hparam.get("save_single_model", False)
        if isinstance(save_single_model, str):
            save_single_model = save_single_model.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            save_single_model = bool(save_single_model)
        if save_single_model:
            checkpoint_path = self.save_single_checkpoint()
            print(f"Saved single model checkpoint: {checkpoint_path}")

        tta_enabled = self.hparam.get("tta_eval", False)
        if isinstance(tta_enabled, str):
            tta_enabled = tta_enabled.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            tta_enabled = bool(tta_enabled)
        if tta_enabled:
            from .tta import run_tta_comparison

            tta_rows = run_tta_comparison(
                self,
                os.path.join(self.hparam["data_path"], "tta"),
            )
            if tta_rows:
                print("TTA comparison")
                for row in tta_rows:
                    print(
                        f"{row['mode']}: "
                        f"entropy {row['entropy_before']:.6f} -> "
                        f"{row['entropy_after']:.6f}; "
                        f"delta={row['entropy_delta']:.6f}"
                    )
        self.transmit_model()

    def save_model(self, num_epoch):
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['id']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

    def _single_checkpoint_path(self):
        configured = str(self.hparam.get("checkpoint_file", "") or "").strip()
        if configured:
            return configured
        return os.path.join(
            self.hparam["data_path"],
            "checkpoint",
            "model.pt",
        )

    def save_single_checkpoint(self):
        path = self._single_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = OrderedDict(
            (key, value.detach().cpu())
            for key, value in self.model.state_dict().items()
        )
        omega_by_key = getattr(self, "_fedcrmf_last_omega_by_key", None)
        if isinstance(omega_by_key, dict):
            omega_by_key = {
                key: value.detach().cpu()
                for key, value in omega_by_key.items()
                if isinstance(value, torch.Tensor)
            }
        else:
            omega_by_key = {}
        payload = {
            "model_state_dict": model_state,
            "fedcrmf_last_omega_by_key": omega_by_key,
            "round": int(getattr(self, "_round", 0)),
            "id": self.hparam.get("id", ""),
            "variant_name": self.hparam.get("variant_name", ""),
            "server_method": self.hparam.get("server_method", ""),
            "client_method": self.hparam.get("client_method", ""),
            "split_scheme": self.hparam.get("split_scheme", ""),
            "freeze_bn": self.hparam.get("freeze_bn", ""),
            "fedcrmf_history_length": self.hparam.get("fedcrmf_history_length", ""),
            "fedcrmf_mu": self.hparam.get("fedcrmf_mu", ""),
        }
        torch.save(payload, path)
        return path

    def load_single_checkpoint(self, path=None):
        checkpoint_path = path or self._single_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Single model checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_state = payload["model_state_dict"]
            omega_by_key = payload.get("fedcrmf_last_omega_by_key", payload.get("core_last_omega_by_key", {}))
        else:
            model_state = payload
            omega_by_key = {}
        self.model.load_state_dict(model_state)
        if isinstance(omega_by_key, dict):
            self._fedcrmf_last_omega_by_key = {
                key: value.detach().cpu()
                for key, value in omega_by_key.items()
                if isinstance(value, torch.Tensor)
            }
        print(f"Loaded single model checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _update_pairwise_client_update_metrics(self, last_weights, local_states):
        if bool(self.hparam.get("disable_pairwise_update_metrics", False)):
            self._last_round_extra_metrics["mean_pairwise_client_update_cosine"] = "N/A"
            self._last_round_extra_metrics["std_pairwise_client_update_cosine"] = "N/A"
            self._last_round_extra_metrics["min_pairwise_client_update_cosine"] = "N/A"
            return

        num_clients = len(local_states)
        if num_clients < 2:
            self._last_round_extra_metrics["mean_pairwise_client_update_cosine"] = "N/A"
            self._last_round_extra_metrics["std_pairwise_client_update_cosine"] = "N/A"
            self._last_round_extra_metrics["min_pairwise_client_update_cosine"] = "N/A"
            return

        dot_products = torch.zeros((num_clients, num_clients), dtype=torch.float64)
        squared_norms = torch.zeros(num_clients, dtype=torch.float64)
        for key, global_param in last_weights.items():
            if not torch.is_floating_point(global_param):
                continue
            flat_deltas = torch.stack(
                [(state[key] - global_param).reshape(-1).to(torch.float64) for state in local_states],
                dim=0,
            )
            if flat_deltas.numel() == 0:
                continue
            dot_products += flat_deltas @ flat_deltas.t()
            squared_norms += (flat_deltas ** 2).sum(dim=1)

        norm_products = torch.sqrt(torch.outer(squared_norms, squared_norms)).clamp_min(1e-12)
        cosine_matrix = dot_products / norm_products
        upper_idx = torch.triu_indices(num_clients, num_clients, offset=1)
        cosine_values = cosine_matrix[upper_idx[0], upper_idx[1]]
        if cosine_values.numel() == 0:
            mean_cos = std_cos = min_cos = "N/A"
        else:
            mean_cos = float(cosine_values.mean().item())
            std_cos = float(cosine_values.std(unbiased=False).item())
            min_cos = float(cosine_values.min().item())
        self._last_round_extra_metrics["mean_pairwise_client_update_cosine"] = mean_cos
        self._last_round_extra_metrics["std_pairwise_client_update_cosine"] = std_cos
        self._last_round_extra_metrics["min_pairwise_client_update_cosine"] = min_cos

    def _resolve_metadata_csv_path(self):
        dataset = self.ds_bundle.dataset
        split_scheme = str(self.hparam.get("split_scheme", "official"))
        filename = "metadata.csv" if split_scheme == "official" else f"{split_scheme}.csv"

        data_dir = getattr(dataset, "_data_dir", None)
        if data_dir is not None:
            candidate = Path(data_dir) / filename
            if candidate.exists():
                return candidate

        if str(self.hparam.get("dataset", "")).lower() == "pacs":
            candidate = Path(__file__).resolve().parents[1] / "resources" / "pacs_v1.0" / filename
            if candidate.exists():
                return candidate
        return None

    def _read_split_domains(self):
        metadata_path = self._resolve_metadata_csv_path()
        split_domains = {}
        if metadata_path is None:
            return split_domains
        with open(metadata_path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "split" not in reader.fieldnames or "domain" not in reader.fieldnames:
                return split_domains
            for row in reader:
                split = row.get("split")
                domain = row.get("domain")
                if not split or not domain:
                    continue
                split_domains.setdefault(split, set()).add(domain)
        return {k: sorted(v) for k, v in split_domains.items()}

    def _join_domain_names(self, names):
        if not names:
            return "N/A"
        return ", ".join(names)

    def _infer_variant_name(self, server_name, client_name):
        if self.hparam.get("variant_name"):
            return self.hparam["variant_name"]
        if server_name in {"DGPM", "FedDGPM"}:
            return "full"
        return "baseline"

    def save_record(
        self,
        metric_dict,
        metrics_per_round,
        best_id_round,
        best_id_val,
        best_id_test,
        best_lodo_round,
        best_lodo_val,
        best_lodo_test,
    ):
        # Persist one human-readable record file per run in the experiment output directory.
        data_path = self.hparam['data_path']
        os.makedirs(data_path, exist_ok=True)
        record_path = os.path.join(data_path, "record")

        final_round = self.num_rounds - 1 if self.num_rounds > 0 else 0
        key_metric = self.ds_bundle.key_metric

        def extract(split_name):
            split_metric = metric_dict.get(split_name, {})
            if isinstance(split_metric, dict):
                return split_metric.get(key_metric, "N/A")
            return "N/A"

        def metrics_at_round(round_idx):
            for round_item in metrics_per_round:
                if len(round_item) >= 2 and round_item[0] == round_idx:
                    return round_item[1]
            return {}

        def extract_from(round_metrics, split_name):
            split_metric = round_metrics.get(split_name, {})
            if isinstance(split_metric, dict):
                return split_metric.get(key_metric, "N/A")
            return "N/A"

        val_score = extract("val")
        test_score = extract("test")
        id_val_score = extract("id_val")
        id_test_score = extract("id_test")

        id_selected_metrics = metrics_at_round(best_id_round)
        has_lodo_val = any(
            "val" in round_item[1]
            for round_item in metrics_per_round
            if len(round_item) >= 2
        )
        lodo_selected_metrics = (
            metrics_at_round(best_lodo_round) if has_lodo_val else {}
        )

        id_selected_lodo_val = extract_from(id_selected_metrics, "val")
        id_selected_lodo_test = extract_from(id_selected_metrics, "test")
        id_selected_id_val = extract_from(id_selected_metrics, "id_val")
        id_selected_id_test = extract_from(id_selected_metrics, "id_test")

        lodo_selected_lodo_val = extract_from(lodo_selected_metrics, "val")
        lodo_selected_lodo_test = extract_from(lodo_selected_metrics, "test")
        lodo_selected_id_val = extract_from(lodo_selected_metrics, "id_val")
        lodo_selected_id_test = extract_from(lodo_selected_metrics, "id_test")
        reported_best_lodo_round = best_lodo_round if has_lodo_val else "N/A"
        reported_best_lodo_val = best_lodo_val if has_lodo_val else "N/A"
        reported_best_lodo_test = best_lodo_test if has_lodo_val else "N/A"

        client_name = self.clients[0].name if len(self.clients) > 0 else "N/A"
        server_name = self.__class__.__name__
        split_scheme = self.hparam.get("split_scheme", "official")
        split_domains = self._read_split_domains()
        target_domain_name = self._join_domain_names(split_domains.get("test") or split_domains.get("val"))
        validation_domain_name = self._join_domain_names(split_domains.get("val"))
        source_domain_name = self._join_domain_names(split_domains.get("train"))
        selection_metric_name = (
            f"id_val/{key_metric}" if "id_val" in metric_dict else f"val/{key_metric}"
        )
        if "id_val" in metric_dict:
            selected_round_by_validation = best_id_round
            selected_validation_score = best_id_val
            selected_lodo_test_by_validation = id_selected_lodo_test
            selected_id_test_by_validation = best_id_test
        else:
            selected_round_by_validation = best_lodo_round
            selected_validation_score = best_lodo_val
            selected_lodo_test_by_validation = best_lodo_test
            selected_id_test_by_validation = "N/A"
        variant_name = self._infer_variant_name(server_name, client_name)

        lines = [
            f"Experiment ID: {self.hparam.get('id', 'N/A')}",
            f"Dataset: {self.hparam.get('dataset', 'N/A')}",
            f"backbone: {self.hparam.get('backbone', 'N/A')}",
            f"freeze_bn: {self.hparam.get('freeze_bn', 'N/A')}",
            f"Method: {server_name} (Server) + {client_name} (Client)",
            f"experiment_protocol: {self.hparam.get('experiment_protocol', 'legacy')}",
            f"implementation_revision: {self.hparam.get('implementation_revision', 'legacy')}",
            f"split_scheme: {split_scheme}",
            f"loaded_split_scheme: {self.hparam.get('loaded_split_scheme', 'N/A')}",
            f"protocol_validation_status: {self.hparam.get('protocol_validation_status', 'N/A')}",
            f"loaded_split_domain_ids: {self.hparam.get('loaded_split_domain_ids', 'N/A')}",
            f"loaded_client_domain_ids: {self.hparam.get('loaded_client_domain_ids', 'N/A')}",
            f"eval_exclude_splits: {self.hparam.get('eval_exclude_splits', [])}",
            f"target_domain_name: {target_domain_name}",
            f"validation_domain_name: {validation_domain_name}",
            f"source_domain_name: {source_domain_name}",
            f"seed: {self.hparam.get('seed', 'N/A')}",
            f"cudnn_benchmark: {self.hparam.get('cudnn_benchmark', 'N/A')}",
            f"cudnn_deterministic: {self.hparam.get('cudnn_deterministic', 'N/A')}",
            f"variant_name: {variant_name}",
            f"Clients: {self.hparam.get('num_clients', 'N/A')} (iid={self.hparam.get('iid', 'N/A')})",
            f"clients_per_domain: {self.hparam.get('clients_per_domain', 'N/A')}",
            f"pacs_total_domain_subclients: {self.hparam.get('pacs_total_domain_subclients', 'N/A')}",
            f"Client fraction: {self.hparam.get('fraction', 'N/A')}",
            f"Rounds: {self.num_rounds} (round 0-{final_round})",
            f"Local epochs: {self.hparam.get('local_epochs', 'N/A')}",
            f"Batch size: {self.hparam.get('batch_size', 'N/A')}",
            f"Learning rate: {self.hparam.get('lr', 'N/A')}",
            f"Optimizer: {self.hparam.get('optimizer', 'N/A')}",
            f"Weight decay: {self.hparam.get('weight_decay', 'N/A')}",
            f"selection_metric_name: {selection_metric_name}",
            f"selected_round_by_validation: {selected_round_by_validation}",
            f"selected_validation_score: {selected_validation_score}",
            f"selected_lodo_test_by_validation: {selected_lodo_test_by_validation}",
            f"selected_id_test_by_validation: {selected_id_test_by_validation}",
            f"selected_test_by_validation: {selected_lodo_test_by_validation}",
            "",
            "Selection details",
            f"ID-val selection: round {best_id_round}, id_val = {best_id_val}, lodo_test = {id_selected_lodo_test}, id_test = {id_selected_id_test}",
            f"LODO-val selection: round {reported_best_lodo_round}, lodo_val = {reported_best_lodo_val}, lodo_test = {reported_best_lodo_test}, id_test = {lodo_selected_id_test}",
            "",
            f"Final round ({final_round})",
            f"LODO val {key_metric}: {val_score}",
            f"LODO test {key_metric}: {test_score}",
            f"ID val {key_metric}: {id_val_score}",
            f"ID test {key_metric}: {id_test_score}",
            "",
            "Best results (program record)",
            f"Best ID: round {best_id_round}, id_val = {best_id_val}, id_test = {best_id_test}, lodo_test = {id_selected_lodo_test}",
            f"Best LODO: round {reported_best_lodo_round}, test = {reported_best_lodo_test}",
        ]

        # Per-round metrics summary (one line per round).
        if metrics_per_round:
            extra_metric_keys = []
            for round_item in metrics_per_round:
                if len(round_item) >= 3 and isinstance(round_item[2], dict):
                    for k in round_item[2].keys():
                        if k not in extra_metric_keys:
                            extra_metric_keys.append(k)
            header = "round\tval\t\ttest\t\tid_val\t\tid_test"
            if extra_metric_keys:
                header += "\t" + "\t".join(extra_metric_keys)
            lines.extend([
                "",
                "Per-round metrics",
                header,
            ])
            for round_item in metrics_per_round:
                round_idx = round_item[0]
                round_metrics = round_item[1]
                round_extra = round_item[2] if (len(round_item) >= 3 and isinstance(round_item[2], dict)) else {}
                val_m = round_metrics.get("val", {}).get(key_metric, "N/A")
                test_m = round_metrics.get("test", {}).get(key_metric, "N/A")
                id_val_m = round_metrics.get("id_val", {}).get(key_metric, "N/A")
                id_test_m = round_metrics.get("id_test", {}).get(key_metric, "N/A")
                row = f"{round_idx}\t{val_m}\t{test_m}\t{id_val_m}\t{id_test_m}"
                if extra_metric_keys:
                    row += "".join(f"\t{round_extra.get(k, 'N/A')}" for k in extra_metric_keys)
                lines.append(row)

        if self._last_round_layer_metrics:
            lines.extend([
                "",
                "Per-layer diagnostics (final round)",
                "layer_name\toutlier_ratio\tdrop_ratio",
            ])
            for layer_name, layer_metric in self._last_round_layer_metrics.items():
                outlier_ratio = layer_metric.get("outlier_ratio", "N/A")
                drop_ratio = layer_metric.get("drop_ratio", "N/A")
                lines.append(f"{layer_name}\t{outlier_ratio}\t{drop_ratio}")

        # Record DGPM-specific parameters for reproducibility.
        if server_name in {"DGPM", "FedDGPM"}:
            lines.extend([
                "",
                "DGPM params",
                f"dgpm_zscore_threshold: {self.hparam.get('dgpm_zscore_threshold', 'N/A')}",
                f"dgpm_warmup_rounds: {self.hparam.get('dgpm_warmup_rounds', 'N/A')}",
                f"dgpm_ema_beta: {self.hparam.get('dgpm_ema_beta', 'N/A')}",
                f"dgpm_stable_outlier_prob_threshold: {self.hparam.get('dgpm_stable_outlier_prob_threshold', 'N/A')}",
                f"dgpm_stable_drop_max_ratio: {self.hparam.get('dgpm_stable_drop_max_ratio', 'N/A')}",
                f"dgpm_hist_keys: {self.hparam.get('dgpm_hist_keys', 'classifier,fc')}",
                f"dgpm_hist_max_numel: {self.hparam.get('dgpm_hist_max_numel', 2000000)}",
            ])





        if server_name == "FedCRMFServer":
            lines.extend([
                "",
                "FedCRMF params",
                f"fedcrmf_history_length: {self.hparam.get('fedcrmf_history_length', 'N/A')}",
                f"fedcrmf_warmup_rounds: {self.hparam.get('fedcrmf_warmup_rounds', 'N/A')}",
                f"fedcrmf_mu: {self.hparam.get('fedcrmf_mu', 'N/A')}",
                f"fedcrmf_gate_variant: {self.hparam.get('fedcrmf_gate_variant', 'full')}",
                f"fedcrmf_gate_dropout_p: {self.hparam.get('fedcrmf_gate_dropout_p', 'N/A')}",
                f"fedcrmf_alpha_mode: {self.hparam.get('fedcrmf_alpha_mode', 'N/A')}",
                f"fedcrmf_hist_keys: {self.hparam.get('fedcrmf_hist_keys', 'ALL')}",
                f"fedcrmf_hist_max_numel: {self.hparam.get('fedcrmf_hist_max_numel', 5000000)}",
                "fedcrmf_filter: none",
                "fedcrmf_score: ||P_d R||_{F,alpha} / sqrt(L)",
                "fedcrmf_gate_target: complete current FedAvg coordinate update",
                "fedcrmf_closed_form: delta = u / (1 + mu * score)",
            ])

        if client_name == "FedSR":
            lines.extend([
                "",
                "FedSR params",
                f"fedsr_l2_regularizer: {self.hparam.get('fedsr_l2_regularizer', 'N/A')}",
                f"fedsr_cmi_regularizer: {self.hparam.get('fedsr_cmi_regularizer', 'N/A')}",
            ])

        if server_name == "FedIIRServer":
            lines.extend([
                "",
                "FedIIR params",
                f"fediir_penalty: {self.hparam.get('fediir_penalty', 'N/A')}",
                f"fediir_ema: {self.hparam.get('fediir_ema', 'N/A')}",
            ])




        with open(record_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        def json_ready(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().item() if value.numel() == 1 else str(value)
            return value

        summary = OrderedDict([
            ("experiment_id", self.hparam.get("id", "N/A")),
            ("variant_name", variant_name),
            ("dataset", self.hparam.get("dataset", "N/A")),
            ("backbone", self.hparam.get("backbone", "N/A")),
            ("freeze_bn", self.hparam.get("freeze_bn", "N/A")),
            ("experiment_protocol", self.hparam.get("experiment_protocol", "legacy")),
            ("implementation_revision", self.hparam.get("implementation_revision", "legacy")),
            ("split_scheme", split_scheme),
            ("loaded_split_scheme", self.hparam.get("loaded_split_scheme", "N/A")),
            ("protocol_validation_status", self.hparam.get("protocol_validation_status", "N/A")),
            ("loaded_split_domain_ids", self.hparam.get("loaded_split_domain_ids", "N/A")),
            ("loaded_client_domain_ids", self.hparam.get("loaded_client_domain_ids", "N/A")),
            ("eval_exclude_splits", self.hparam.get("eval_exclude_splits", [])),
            ("method", f"{server_name}+{client_name}"),
            ("seed", self.hparam.get("seed", "N/A")),
            ("cudnn_benchmark", self.hparam.get("cudnn_benchmark", "N/A")),
            ("cudnn_deterministic", self.hparam.get("cudnn_deterministic", "N/A")),
            ("num_clients", self.hparam.get("num_clients", "N/A")),
            ("clients_per_domain", self.hparam.get("clients_per_domain", "N/A")),
            (
                "pacs_total_domain_subclients",
                self.hparam.get("pacs_total_domain_subclients", "N/A"),
            ),
            ("fraction", self.hparam.get("fraction", "N/A")),
            ("iid", self.hparam.get("iid", "N/A")),
            ("num_rounds", self.num_rounds),
            ("local_epochs", self.hparam.get("local_epochs", "N/A")),
            ("batch_size", self.hparam.get("batch_size", "N/A")),
            ("lr", self.hparam.get("lr", "N/A")),
            ("optimizer", self.hparam.get("optimizer", "N/A")),
            ("weight_decay", self.hparam.get("weight_decay", "N/A")),
            ("hparam1", self.hparam.get("hparam1", "N/A")),
            ("hparam2", self.hparam.get("hparam2", "N/A")),
            ("hparam3", self.hparam.get("hparam3", "N/A")),
            ("hparam4", self.hparam.get("hparam4", "N/A")),
            ("hparam5", self.hparam.get("hparam5", "N/A")),
            ("fedga_d", self.hparam.get("fedga_d", "N/A")),
            ("fedga_eps", self.hparam.get("fedga_eps", "N/A")),
            ("fedga_init", self.hparam.get("fedga_init", "N/A")),
            ("rmf_history_length", self.hparam.get("rmf_history_length", "N/A")),
            ("rmf_warmup_rounds", self.hparam.get("rmf_warmup_rounds", "N/A")),
            ("rmf_lambda_d", self.hparam.get("rmf_lambda_d", "N/A")),
            ("rmf_lambda_t", self.hparam.get("rmf_lambda_t", "N/A")),
            ("rmf_mu", self.hparam.get("rmf_mu", "N/A")),
            ("rmf_gate_scale", self.hparam.get("rmf_gate_scale", "N/A")),
            ("rmf_omega_min", self.hparam.get("rmf_omega_min", "N/A")),
            ("rmf_alpha_mode", self.hparam.get("rmf_alpha_mode", "N/A")),
            ("rmf_projection_mode", self.hparam.get("rmf_projection_mode", "N/A")),
            ("rmf_gate_mode", self.hparam.get("rmf_gate_mode", "N/A")),
            ("rmf_use_response_weight", self.hparam.get("rmf_use_response_weight", "N/A")),
            ("rmf_hist_keys", self.hparam.get("rmf_hist_keys", "N/A")),
            ("rmf_hist_max_numel", self.hparam.get("rmf_hist_max_numel", "N/A")),
            ("rmg_history_length", self.hparam.get("rmg_history_length", "N/A")),
            ("rmg_warmup_rounds", self.hparam.get("rmg_warmup_rounds", "N/A")),
            ("rmg_mu", self.hparam.get("rmg_mu", "N/A")),
            ("rmg_gate_scale", self.hparam.get("rmg_gate_scale", "N/A")),
            ("rmg_omega_min", self.hparam.get("rmg_omega_min", "N/A")),
            ("rmg_alpha_mode", self.hparam.get("rmg_alpha_mode", "N/A")),
            ("rmg_hist_keys", self.hparam.get("rmg_hist_keys", "N/A")),
            ("rmg_hist_max_numel", self.hparam.get("rmg_hist_max_numel", "N/A")),
            ("fedcrmf_history_length", self.hparam.get("fedcrmf_history_length", "N/A")),
            ("fedcrmf_warmup_rounds", self.hparam.get("fedcrmf_warmup_rounds", "N/A")),
            ("fedcrmf_mu", self.hparam.get("fedcrmf_mu", "N/A")),
            ("fedcrmf_alpha_mode", self.hparam.get("fedcrmf_alpha_mode", "N/A")),
            ("fedcrmf_hist_keys", self.hparam.get("fedcrmf_hist_keys", "N/A")),
            ("fedcrmf_hist_max_numel", self.hparam.get("fedcrmf_hist_max_numel", "N/A")),
            ("fedsr_l2_regularizer", self.hparam.get("fedsr_l2_regularizer", "N/A")),
            ("fedsr_cmi_regularizer", self.hparam.get("fedsr_cmi_regularizer", "N/A")),
            ("fediir_penalty", self.hparam.get("fediir_penalty", "N/A")),
            ("fediir_ema", self.hparam.get("fediir_ema", "N/A")),
            ("fedprox_mu", self.hparam.get("fedprox_mu", "N/A")),
            ("fedomg_global_lr", self.hparam.get("fedomg_global_lr", "N/A")),
            ("fedomg_search_radius", self.hparam.get("fedomg_search_radius", "N/A")),
            ("fedomg_solver_lr", self.hparam.get("fedomg_solver_lr", "N/A")),
            ("fedomg_solver_momentum", self.hparam.get("fedomg_solver_momentum", "N/A")),
            ("fedomg_solver_iterations", self.hparam.get("fedomg_solver_iterations", "N/A")),
            ("fedga_step_size", self.hparam.get("fedga_step_size", "N/A")),
            ("fedga_metric", self.hparam.get("fedga_metric", "N/A")),
            ("selection_metric_name", selection_metric_name),
            ("selected_round_by_validation", selected_round_by_validation),
            ("selected_validation_score", selected_validation_score),
            ("selected_lodo_test_by_validation", selected_lodo_test_by_validation),
            ("selected_id_test_by_validation", selected_id_test_by_validation),
            ("selected_test_by_validation", selected_lodo_test_by_validation),
            ("id_val_selected_round", best_id_round),
            ("id_val_selected_id_val", best_id_val),
            ("id_val_selected_lodo_val", id_selected_lodo_val),
            ("id_val_selected_lodo_test", id_selected_lodo_test),
            ("id_val_selected_id_test", id_selected_id_test),
            ("lodo_val_selected_round", reported_best_lodo_round),
            ("lodo_val_selected_lodo_val", lodo_selected_lodo_val),
            ("lodo_val_selected_lodo_test", lodo_selected_lodo_test),
            ("lodo_val_selected_id_val", lodo_selected_id_val),
            ("lodo_val_selected_id_test", lodo_selected_id_test),
            ("final_round", final_round),
            ("final_lodo_test", test_score),
            ("final_id_test", id_test_score),
            ("best_lodo_round", reported_best_lodo_round),
            ("best_lodo_test", reported_best_lodo_test),
            ("best_id_round", best_id_round),
            ("best_id_test", best_id_test),
        ])
        summary = OrderedDict((k, json_ready(v)) for k, v in summary.items())
        summary_line = (
            f"{summary['variant_name']} | {summary['method']} | "
            f"{summary['dataset']} {summary['split_scheme']} | "
            f"selected_round={summary['selected_round_by_validation']} | "
            f"selected_val={summary['selected_validation_score']} | "
            f"selected_lodo_test={summary['selected_lodo_test_by_validation']} | "
            f"selected_id_test={summary['selected_id_test_by_validation']} | "
            f"final_lodo_test={summary['final_lodo_test']} | "
            f"best_lodo_test={summary['best_lodo_test']}"
        )
        summary["summary_line"] = summary_line

        with open(os.path.join(data_path, "summary_result.txt"), "w", encoding="utf-8") as fh:
            fh.write(summary_line + "\n")
        with open(os.path.join(data_path, "summary_result.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

        # Maintain one compact result table in the parent experiment folder.
        # Example: outputs/pacs/resnet50_allparams/summary_all_results.csv
        group_dir = os.path.dirname(os.path.normpath(data_path))
        aggregate_path = os.path.join(group_dir, "summary_all_results.csv")
        aggregate_fields = [
            "variant_name",
            "method",
            "seed",
            "selected_round",
            "selected_lodo_test",
            "final_lodo_test",
            "best_lodo_test",
            "experiment_id",
        ]
        aggregate_row = OrderedDict([
            ("variant_name", summary["variant_name"]),
            ("method", summary["method"]),
            ("seed", summary["seed"]),
            ("selected_round", summary["selected_round_by_validation"]),
            ("selected_lodo_test", summary["selected_lodo_test_by_validation"]),
            ("final_lodo_test", summary["final_lodo_test"]),
            ("best_lodo_test", summary["best_lodo_test"]),
            ("experiment_id", summary["experiment_id"]),
        ])

        existing_rows = []
        if os.path.exists(aggregate_path):
            with open(aggregate_path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("experiment_id") != str(aggregate_row["experiment_id"]):
                        existing_rows.append(row)
        existing_rows.append({k: str(aggregate_row.get(k, "")) for k in aggregate_fields})
        existing_rows.sort(key=lambda row: row.get("variant_name", ""))
        with open(aggregate_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=aggregate_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
