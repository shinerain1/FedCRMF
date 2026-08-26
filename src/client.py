import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import wandb
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_train_loader


class ERM:
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        self.client_id = client_id
        self.device = device
        self.dataset = dataset
        self.ds_bundle = ds_bundle
        self.hparam = hparam
        self.local_epochs = hparam["local_epochs"]
        self.batch_size = hparam["batch_size"]
        self.num_workers = int(hparam.get("num_workers", 0))
        self.pin_memory = bool(hparam.get("pin_memory", False)) and getattr(device, "type", str(device)) == "cuda"
        self.optimizer_name = hparam["optimizer"]
        self.optim_config = hparam["optimizer_config"]
        self.dataloader = get_train_loader(
            self.loader_type,
            self.dataset,
            batch_size=self.batch_size,
            uniform_over_groups=None,
            grouper=self.ds_bundle.grouper,
            distinct_groups=False,
            n_groups_per_batch=hparam["n_groups_per_batch"],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.save_opt_state = bool(hparam.get("save_opt_state", False))
        base_path = os.path.abspath(hparam.get("data_path", "."))
        self.opt_dict_path = os.path.join(base_path, "opt_dict", f"client_{client_id}.pt")
        self.sch_dict_path = os.path.join(base_path, "sch_dict", f"client_{client_id}.pt")

    @property
    def loader_type(self):
        return "standard"

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer,
            factor=1,
            total_iters=1,
        )

    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        del self.scheduler, self.optimizer
        if getattr(self.device, "type", str(self.device)) == "cuda":
            torch.cuda.empty_cache()

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        metadata = metadata.to(self.device)
        outputs = self.model(x)
        return {
            "y_true": y_true,
            "y_pred": outputs,
            "metadata": metadata,
        }

    def step(self, results):
        loss = self.ds_bundle.loss.compute(
            results["y_pred"],
            results["y_true"],
            return_dict=False,
        )
        objective = loss.mean()
        total_loss = loss.sum().item()
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss

    def fit(self, server_round):
        self.init_train()
        training_loss = 0.0
        for epoch in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                training_loss += self.step(self.process_batch(batch))
            if self.hparam.get("wandb", False):
                wandb.log(
                    {f"loss/{self.client_id}": training_loss / len(self.dataset)},
                    step=server_round * self.local_epochs + epoch,
                )
        self.end_train()


class FedProx(ERM):
    """FedProx client with a proximal penalty around the received model."""

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.proximal_mu = float(hparam.get("fedprox_mu", 0.1))
        self.microbatch_size = int(
            hparam.get("fedprox_microbatch_size", self.batch_size)
        )
        if self.microbatch_size < 1:
            raise ValueError("fedprox_microbatch_size must be >= 1")
        self._global_parameters = None

    def init_train(self):
        super().init_train()
        self._global_parameters = tuple(
            parameter.detach().clone() for parameter in self.model.parameters()
        )

    def _proximal_term(self):
        return sum(
            (parameter - global_parameter).square().sum()
            for parameter, global_parameter in zip(
                self.model.parameters(), self._global_parameters
            )
        )

    def fit(self, server_round):
        self.init_train()
        training_loss = 0.0
        for epoch in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                x, y_true, metadata = batch
                num_examples = int(y_true.shape[0])
                self.optimizer.zero_grad(set_to_none=True)
                erm_loss_sum = 0.0
                for start in range(0, num_examples, self.microbatch_size):
                    stop = min(start + self.microbatch_size, num_examples)
                    results = self.process_batch(
                        (x[start:stop], y_true[start:stop], metadata[start:stop])
                    )
                    microbatch_loss = self.ds_bundle.loss.compute(
                        results["y_pred"],
                        results["y_true"],
                        return_dict=False,
                    ).sum()
                    (microbatch_loss / num_examples).backward()
                    erm_loss_sum += microbatch_loss.detach().item()
                proximal_term = self._proximal_term()
                proximal_objective = 0.5 * self.proximal_mu * proximal_term
                proximal_objective.backward()
                self.optimizer.step()
                training_loss += (
                    erm_loss_sum + num_examples * proximal_objective.detach().item()
                )
            if self.hparam.get("wandb", False):
                wandb.log(
                    {f"loss/{self.client_id}": training_loss / len(self.dataset)},
                    step=server_round * self.local_epochs + epoch,
                )
        self.end_train()

    def end_train(self):
        self._global_parameters = None
        super().end_train()


class FedSR(ERM):
    """Federated stochastic representation learning client."""

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.l2_regularizer = float(
            hparam.get("fedsr_l2_regularizer", hparam.get("hparam1", 1e-3))
        )
        self.cmi_regularizer = float(
            hparam.get("fedsr_cmi_regularizer", hparam.get("hparam2", 1e-4))
        )
        self._reference_params_state = None

    def setup_model(self, featurizer, classifier):
        super().setup_model(featurizer, classifier)
        self._reference_params_state = torch.ones(
            self.ds_bundle.n_classes,
            2 * self._featurizer.n_outputs,
        )

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.reference_params = nn.Parameter(
            self._reference_params_state.to(self.device)
        )
        self.optimizer = eval(self.optimizer_name)(
            [*self.model.parameters(), self.reference_params],
            **self.optim_config,
        )

    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self._reference_params_state = self.reference_params.detach().cpu()
        self.model.to("cpu")
        del self.reference_params, self.optimizer
        if getattr(self.device, "type", str(self.device)) == "cuda":
            torch.cuda.empty_cache()

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        metadata = metadata.to(self.device)
        feature_params = self.featurizer(x)
        z_dim = feature_params.shape[-1] // 2
        z_mu = feature_params[..., :z_dim]
        z_sigma = F.softplus(feature_params[..., z_dim:]).clamp_min(1e-8)
        features = dist.Independent(dist.Normal(z_mu, z_sigma), 1).rsample()
        return {
            "y_true": y_true,
            "y_pred": self.classifier(features),
            "metadata": metadata,
            "features": features,
            "z_mu": z_mu,
            "z_sigma": z_sigma,
        }

    @staticmethod
    def _l2_penalty(features):
        return features.square().sum() / features.shape[0]

    def _cmi_penalty(self, labels, z_mu, z_sigma):
        dimension = self.reference_params.shape[1] // 2
        target_mu = self.reference_params[labels.long(), :dimension]
        target_sigma = F.softplus(
            self.reference_params[labels.long(), dimension:]
        ).clamp_min(1e-8)
        divergence = (
            torch.log(target_sigma)
            - torch.log(z_sigma)
            + (z_sigma.square() + (target_mu - z_mu).square())
            / (2.0 * target_sigma.square())
            - 0.5
        )
        return divergence.sum() / labels.shape[0]

    def step(self, results):
        erm_loss = self.ds_bundle.loss.compute(
            results["y_pred"],
            results["y_true"],
            return_dict=False,
        ).mean()
        objective = (
            erm_loss
            + self.l2_regularizer * self._l2_penalty(results["features"])
            + self.cmi_regularizer
            * self._cmi_penalty(
                results["y_true"],
                results["z_mu"],
                results["z_sigma"],
            )
        )
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return results["y_true"].shape[0] * objective.item()


class FedIIR(ERM):
    """FedIIR local objective with a server-provided mean gradient."""

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.penalty_weight = float(hparam.get("fediir_penalty", 1e-3))
        self._grad_mean = None

    def set_grad_mean(self, grad_mean):
        self._grad_mean = tuple(grad.detach().cpu() for grad in grad_mean)

    def step(self, results):
        if self._grad_mean is None:
            raise RuntimeError("FedIIR mean gradient was not provided by the server")
        erm_loss = self.ds_bundle.loss.compute(
            results["y_pred"],
            results["y_true"],
            return_dict=False,
        ).mean()
        client_grads = torch.autograd.grad(
            erm_loss,
            tuple(self.classifier.parameters()),
            create_graph=True,
        )
        penalty = sum(
            (client_grad - mean_grad.to(client_grad.device)).square().sum()
            for client_grad, mean_grad in zip(client_grads, self._grad_mean)
        )
        objective = erm_loss + self.penalty_weight * penalty
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return results["y_true"].shape[0] * objective.item()
