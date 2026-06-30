import numpy as np
import torch
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset


class NonIIDSplitter:
    def __init__(self, num_shards, iid, seed):
        self.num_shards = int(num_shards)
        self.iid = float(iid)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def split(self, dataset, domain_field, transform=None):
        domain_field = dataset._metadata_fields.index(domain_field[0])
        metadata = dataset.metadata_array
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.detach().cpu().numpy()
        num_examples_per_domain = np.bincount(metadata[:, domain_field])
        num_domains = len(num_examples_per_domain)
        non_empty = num_examples_per_domain > 0
        main_shards_per_domain = non_empty.astype(int)
        while np.sum(main_shards_per_domain) < self.num_shards:
            ratio = np.divide(
                num_examples_per_domain.astype(float),
                main_shards_per_domain.astype(float),
                out=np.zeros_like(num_examples_per_domain.astype(float)),
                where=main_shards_per_domain != 0,
            )
            main_shards_per_domain[int(np.argmax(ratio))] += 1

        main_domain_per_shard = []
        for domain, count in enumerate(main_shards_per_domain):
            main_domain_per_shard.extend([domain] * int(count))
        num_examples_per_shards = []
        main_ratio = np.array([1 / u if u else 0 for u in main_shards_per_domain])
        non_main_ratio = 1 / self.num_shards
        for main_domain in main_domain_per_shard:
            onehot = np.zeros(len(main_shards_per_domain))
            onehot[main_domain] = 1
            num_examples_per_shards.append(
                num_examples_per_domain
                * (main_ratio * onehot * (1 - self.iid) + non_main_ratio * self.iid)
            )
        float_counts = np.array(num_examples_per_shards)
        int_counts = float_counts.astype(int)
        diff = np.rint(np.sum(float_counts - int_counts, axis=0)).astype(int)
        diff_mask = np.zeros((self.num_shards, num_domains))
        for col in range(num_domains):
            diff_mask[: diff[col], col] = 1
        final_counts = np.rint(int_counts + diff_mask).astype(int)

        indices_per_domain = []
        for domain in range(num_domains):
            local_positions = np.where(metadata[:, domain_field] == domain)[0]
            indices = np.array(dataset.indices)[local_positions]
            indices_per_domain.append(self.rng.permutation(indices))

        shards = []
        pointer = np.zeros(num_domains, dtype=np.int64)
        for shard in range(self.num_shards):
            shard_indices = []
            for domain in range(num_domains):
                offset = final_counts[shard, domain]
                if offset > 0:
                    shard_indices.extend(
                        indices_per_domain[domain][pointer[domain] : pointer[domain] + offset].tolist()
                    )
                    pointer[domain] += offset
            shards.append(WILDSSubset(dataset.dataset, shard_indices, transform=transform))
        return shards


class DomainBalancedSplitter:
    def __init__(self, shards_per_domain, seed):
        self.shards_per_domain = int(shards_per_domain)
        if self.shards_per_domain < 1:
            raise ValueError("shards_per_domain must be at least 1")
        self.rng = np.random.default_rng(seed)

    def split(self, dataset, domain_field, transform=None):
        if isinstance(domain_field, (list, tuple)):
            domain_field = domain_field[0]
        domain_index = dataset._metadata_fields.index(domain_field)
        metadata = dataset.metadata_array
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.detach().cpu().numpy()
        domains = sorted(int(value) for value in np.unique(metadata[:, domain_index]))
        shards = []
        for domain in domains:
            local_positions = np.where(metadata[:, domain_index] == domain)[0]
            base_indices = np.array(dataset.indices)[local_positions]
            perm = self.rng.permutation(base_indices)
            for split_indices in np.array_split(perm, self.shards_per_domain):
                shards.append(
                    WILDSSubset(
                        dataset.dataset,
                        split_indices.tolist(),
                        transform=transform,
                    )
                )
        return shards
