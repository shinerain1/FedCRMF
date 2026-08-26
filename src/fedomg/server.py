import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..server import FedAvg


class FedOMGServer(FedAvg):
    """Protocol-adapted FedOMG server using the official low-dimensional solver."""

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.global_lr = float(hparam.get("fedomg_global_lr", 0.05))
        self.search_radius = float(hparam.get("fedomg_search_radius", 0.5))
        self.solver_lr = float(hparam.get("fedomg_solver_lr", 25.0))
        self.solver_momentum = float(hparam.get("fedomg_solver_momentum", 0.5))
        self.solver_iterations = int(hparam.get("fedomg_solver_iterations", 21))

    def _solve_invariant_update(self, updates):
        num_clients = updates.shape[0]
        gram = updates.mm(updates.t()).to(torch.float64)
        scale = (torch.diag(gram) + 1e-4).sqrt().mean().clamp_min(1e-12)
        normalized_gram = gram / scale.square()
        reference_inner = normalized_gram.mean(dim=1, keepdim=True)
        reference_norm_sq = reference_inner.mean().reshape(1, 1)

        logits = torch.zeros(num_clients, 1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD(
            [logits], lr=self.solver_lr, momentum=self.solver_momentum
        )
        radius = (reference_norm_sq + 1e-4).sqrt() * self.search_radius
        best_logits = logits.detach().clone()
        best_objective = np.inf
        for iteration in range(self.solver_iterations):
            optimizer.zero_grad()
            weights = torch.softmax(logits, dim=0)
            weighted_norm = (
                weights.t().mm(normalized_gram).mm(weights) + 1e-4
            ).sqrt()
            objective = weights.t().mm(reference_inner) + radius * weighted_norm
            objective_value = float(objective.item())
            if objective_value < best_objective:
                best_objective = objective_value
                best_logits = logits.detach().clone()
            if iteration + 1 < self.solver_iterations:
                objective.backward()
                optimizer.step()

        weights = torch.softmax(best_logits, dim=0)
        weighted_norm = (
            weights.t().mm(normalized_gram).mm(weights) + 1e-4
        ).sqrt()
        multiplier = radius.reshape(-1) / (weighted_norm.reshape(-1) + 1e-4)
        coefficients = (1.0 / num_clients + weights * multiplier).reshape(-1)
        coefficients = coefficients / (1.0 + self.search_radius**2)
        invariant_update = (
            coefficients.to(dtype=updates.dtype).unsqueeze(1) * updates
        ).sum(dim=0)
        return invariant_update, weights.reshape(-1), best_objective

    def aggregate(self, sampled_client_indices, coefficients):
        self.model.to("cpu")
        global_vector = parameters_to_vector(self.model.parameters()).detach().float()
        client_updates = []
        for position, client_index in enumerate(sampled_client_indices):
            self.clients[client_index].model.to("cpu")
            local_vector = parameters_to_vector(
                self.clients[client_index].model.parameters()
            ).detach().float()
            client_updates.append(
                (local_vector - global_vector) * float(coefficients[position])
            )
        updates = torch.stack(client_updates, dim=0)
        invariant_update, solver_weights, objective = self._solve_invariant_update(
            updates
        )
        updated_vector = global_vector + self.global_lr * invariant_update
        vector_to_parameters(updated_vector, self.model.parameters())
        self._last_round_extra_metrics.update(
            {
                "fedomg_solver_weights": [float(value) for value in solver_weights],
                "fedomg_solver_objective": float(objective),
                "fedomg_update_norm": float(invariant_update.norm().item()),
            }
        )
