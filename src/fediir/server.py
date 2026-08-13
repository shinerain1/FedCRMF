import torch

from ..server import FedAvg


class FedIIRServer(FedAvg):
    """FedAvg server with FedIIR's EMA classifier-gradient reference."""

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.fediir_ema = float(hparam.get("fediir_ema", 0.95))
        if not 0.0 <= self.fediir_ema < 1.0:
            raise ValueError("fediir_ema must be in [0, 1)")
        self._fediir_grad_mean = None
        self.hparam["fediir_ema"] = self.fediir_ema

    def _compute_mean_grad(self, sampled_client_indices):
        self.model.eval()
        self.model.to(self.device)
        classifier_params = tuple(self.classifier.parameters())
        grad_sum = tuple(torch.zeros_like(param) for param in classifier_params)
        total_batches = 0

        for index in sampled_client_indices:
            for batch in self.clients[index].dataloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad():
                    features = self.featurizer(inputs)
                logits = self.classifier(features)
                loss = self.ds_bundle.loss.compute(
                    logits,
                    labels,
                    return_dict=False,
                ).mean()
                batch_grads = torch.autograd.grad(
                    loss,
                    classifier_params,
                    create_graph=False,
                )
                grad_sum = tuple(
                    accumulated + batch_grad.detach()
                    for accumulated, batch_grad in zip(grad_sum, batch_grads)
                )
                total_batches += 1

        if total_batches == 0:
            raise RuntimeError("FedIIR cannot compute a mean gradient from zero batches")

        current_mean = tuple(
            grad.detach().cpu() / float(total_batches) for grad in grad_sum
        )
        if self._fediir_grad_mean is None:
            previous_mean = tuple(torch.zeros_like(grad) for grad in current_mean)
        else:
            previous_mean = self._fediir_grad_mean
        self._fediir_grad_mean = tuple(
            self.fediir_ema * previous + (1.0 - self.fediir_ema) * current
            for previous, current in zip(previous_mean, current_mean)
        )
        mean_norm = torch.sqrt(
            sum(grad.double().square().sum() for grad in self._fediir_grad_mean)
        )
        self._last_round_extra_metrics["fediir_mean_grad_norm"] = float(
            mean_norm.item()
        )
        self.model.to("cpu")
        if getattr(self.device, "type", str(self.device)) == "cuda":
            torch.cuda.empty_cache()
        return self._fediir_grad_mean

    def train_federated_model(self):
        sampled_client_indices = self.sample_clients()
        self.transmit_model(sampled_client_indices)
        grad_mean = self._compute_mean_grad(sampled_client_indices)
        for index in sampled_client_indices:
            self.clients[index].set_grad_mean(grad_mean)
        selected_total_size = self.update_clients(sampled_client_indices)
        mixing_coefficients = [
            len(self.clients[index]) / selected_total_size
            for index in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)
