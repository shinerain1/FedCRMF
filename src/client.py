import os

import torch
import torch.nn as nn
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
