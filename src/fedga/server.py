from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from ..server import FedAvg


class FedGAServer(FedAvg):
    """FedGA server with source-domain generalization adjustment."""

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.step_size = float(hparam.get("fedga_step_size", 0.2))
        self.metric = str(hparam.get("fedga_metric", "acc")).lower()
        if self.metric not in {"acc", "loss"}:
            raise ValueError("fedga_metric must be 'acc' or 'loss'")
        self.client_val_loaders = []
        self.aggregation_weights = None

    def register_client_val_loaders(self, loaders):
        if len(loaders) != self.num_clients:
            raise ValueError("FedGA requires one source validation loader per client")
        self.client_val_loaders = list(loaders)
        self.aggregation_weights = np.full(
            self.num_clients, 1.0 / self.num_clients, dtype=np.float64
        )

    def _evaluate_model(self, model, dataloader):
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for data, labels, _ in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                logits = model(data)
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_examples += labels.numel()
        model.to("cpu")
        if total_examples == 0:
            raise RuntimeError("FedGA source validation loader is empty")
        return {
            "loss": total_loss / total_examples,
            "acc": total_correct / total_examples,
        }

    def _aggregate_with_weights(self, sampled_client_indices, weights):
        local_states = [
            OrderedDict(
                (key, value.detach().clone().cpu())
                for key, value in self.clients[index].model.state_dict().items()
            )
            for index in sampled_client_indices
        ]
        averaged = OrderedDict()
        q = torch.as_tensor(weights, dtype=torch.float32)
        q = q / q.sum().clamp_min(torch.finfo(q.dtype).eps)
        for key, reference in self.model.state_dict().items():
            if torch.is_floating_point(reference):
                stacked = torch.stack(
                    [state[key].to(torch.float32) for state in local_states], dim=0
                )
                view = q.view(-1, *([1] * (stacked.dim() - 1)))
                averaged[key] = (view * stacked).sum(dim=0).to(reference.dtype)
            else:
                averaged[key] = local_states[0][key]
        self.model.load_state_dict(averaged)

    def _adjust_weights(self, before, after, sampled_client_indices):
        gaps = np.asarray(
            [after[index][self.metric] - before[index][self.metric]
             for index in sampled_client_indices],
            dtype=np.float64,
        )
        max_gap = float(np.max(np.abs(gaps))) if gaps.size else 0.0
        if max_gap <= np.finfo(np.float64).eps:
            return
        signal = -1.0 if self.metric == "acc" else 1.0
        remaining_fraction = max(
            0.0, 1.0 - float(self._round) / max(float(self.num_rounds), 1.0)
        )
        current_step_size = self.step_size * remaining_fraction
        adjustment = signal * (gaps / max_gap) * (
            current_step_size / len(sampled_client_indices)
        )
        for position, client_index in enumerate(sampled_client_indices):
            self.aggregation_weights[client_index] += adjustment[position]
        self.aggregation_weights = np.clip(self.aggregation_weights, 0.0, 1.0)
        total = float(self.aggregation_weights.sum())
        if total <= np.finfo(np.float64).eps:
            self.aggregation_weights.fill(1.0 / self.num_clients)
        else:
            self.aggregation_weights /= total

    def train_federated_model(self):
        if not self.client_val_loaders or self.aggregation_weights is None:
            raise RuntimeError("FedGA source validation loaders were not registered")
        sampled = self.sample_clients()
        self.transmit_model(sampled)
        self.update_clients(sampled)
        before = {
            index: self._evaluate_model(
                self.clients[index].model, self.client_val_loaders[index]
            )
            for index in sampled
        }
        round_weights = np.asarray(
            [self.aggregation_weights[index] for index in sampled], dtype=np.float64
        )
        round_weights /= round_weights.sum()
        self._aggregate_with_weights(sampled, round_weights)
        after = {
            index: self._evaluate_model(self.model, self.client_val_loaders[index])
            for index in sampled
        }
        self._adjust_weights(before, after, sampled)
        self._last_round_extra_metrics.update(
            {
                "fedga_weights": [float(value) for value in self.aggregation_weights],
                "fedga_before": before,
                "fedga_after": after,
            }
        )
