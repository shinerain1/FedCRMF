from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset


class PACS(WILDSDataset):
    _dataset_name = "pacs"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": "",
        }
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
    ):
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (224, 224)
        self._y_type = "long"
        self._y_size = 1
        root_dir = Path(root_dir)
        default_data_dir = root_dir / "pacs_v1.0"
        local_pacs_dir = root_dir / "pacs"
        self._using_local_pacs = local_pacs_dir.exists() and (local_pacs_dir / "images").exists()
        self._data_dir = local_pacs_dir if self._using_local_pacs else default_data_dir

        metadata_filename = "metadata.csv" if split_scheme == "official" else f"{split_scheme}.csv"
        self._n_classes = 7
        metadata_path = self._data_dir / metadata_filename
        if not metadata_path.exists():
            resource_metadata_path = (
                Path(__file__).resolve().parents[1]
                / "resources"
                / "pacs_v1.0"
                / metadata_filename
            )
            if resource_metadata_path.exists():
                metadata_path = resource_metadata_path
            else:
                raise FileNotFoundError(
                    f"Cannot find {metadata_filename} in {self._data_dir} "
                    f"or {resource_metadata_path}."
                )

        df = pd.read_csv(metadata_path)
        self._input_array = df["path"].astype(str).values
        if self._using_local_pacs and len(self._input_array) > 0:
            first_rel_path = Path(self._input_array[0])
            if (
                not (self._data_dir / first_rel_path).exists()
                and (self._data_dir / "images" / first_rel_path).exists()
            ):
                self._input_array = np.array(
                    [str(Path("images") / Path(p)) for p in self._input_array]
                )

        self._split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
            "id_val": 3,
            "id_test": 4,
        }
        self._split_names = {
            "train": "Train",
            "val": "Validation (OOD/Trans)",
            "test": "Test (OOD/Trans)",
            "id_val": "Validation (ID/Cis)",
            "id_test": "Test (ID/Cis)",
        }
        df["split_id"] = df["split"].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(
            np.stack(
                [
                    df["domain_remapped"].values,
                    df["y"].values,
                    np.arange(df["y"].shape[0]),
                ],
                axis=1,
            )
        )
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=["domain"],
        )
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        path = self._data_dir / self._input_array[idx]
        return Image.open(path).convert("RGB")

    def eval(self, y_pred, y_true, metadata):
        correct = (y_pred == y_true).float()
        return {"acc_avg": correct.mean().item()}, f"Average acc: {correct.mean().item():.3f}\n"


class OfficeHome(WILDSDataset):
    _dataset_name = "officehome"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": "",
        }
    }

    _domain_codes = {
        "a": 0,  # Art
        "c": 1,  # Clipart
        "p": 2,  # Product
        "r": 3,  # Real World
    }
    _split_targets = {
        "cpr-a": "a",
        "apr-c": "c",
        "acr-p": "p",
        "acp-r": "r",
    }

    def __init__(
        self,
        version: str = None,
        root_dir: str = "data",
        download: bool = False,
        split_scheme: str = "official",
    ):
        self._version: Optional[str] = version
        self._split_scheme = split_scheme
        self._original_resolution = (224, 224)
        self._y_type = "long"
        self._y_size = 1
        self._n_classes = 65

        root_dir = Path(root_dir)
        candidates = [
            root_dir / "office_home_v1.0",
            root_dir / "officehome_v1.0",
            root_dir / "officehome",
        ]
        self._data_dir = next((path for path in candidates if path.exists()), candidates[0])
        metadata_path = self._data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot find OfficeHome metadata.csv in {self._data_dir}.")

        df = pd.read_csv(metadata_path)
        df = self._apply_lodo_split(df, split_scheme)

        self._input_array = df["path"].astype(str).values
        self._split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
            "id_val": 3,
            "id_test": 4,
        }
        self._split_names = {
            "train": "Train",
            "val": "Validation (OOD/Trans)",
            "test": "Test (OOD/Trans)",
            "id_val": "Validation (ID/Cis)",
            "id_test": "Test (ID/Cis)",
        }
        df["split_id"] = df["split"].apply(lambda x: self._split_dict[x])
        self._split_array = df["split_id"].values
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        self._metadata_fields = ["domain", "y", "idx"]
        self._metadata_array = torch.tensor(
            np.stack(
                [
                    df["domain_remapped"].values,
                    df["y"].values,
                    np.arange(df["y"].shape[0]),
                ],
                axis=1,
            )
        )
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=["domain"],
        )
        super().__init__(root_dir, download, split_scheme)

    def _apply_lodo_split(self, df, split_scheme):
        scheme = str(split_scheme).strip().lower().replace("_", "-")
        if scheme == "official":
            return df.copy()
        if scheme not in self._split_targets:
            valid = ", ".join(sorted(self._split_targets))
            raise ValueError(f"Unsupported OfficeHome split_scheme={split_scheme!r}. Valid: {valid}")

        target_code = self._split_targets[scheme]
        target_id = self._domain_codes[target_code]
        source_ids = sorted(set(self._domain_codes.values()) - {target_id})
        domain = df["domain_remapped"].astype(int)
        rng = np.random.RandomState(2026 + target_id)

        parts = []
        for source_id in source_ids:
            indices = df.index[domain.eq(source_id)].to_numpy()
            rng.shuffle(indices)
            n_total = len(indices)
            n_id_val = max(1, int(round(0.10 * n_total)))
            n_id_test = max(1, int(round(0.10 * n_total)))
            train_indices = indices[: n_total - n_id_val - n_id_test]
            id_val_indices = indices[n_total - n_id_val - n_id_test : n_total - n_id_test]
            id_test_indices = indices[n_total - n_id_test :]
            for new_split, split_indices in [
                ("train", train_indices),
                ("id_val", id_val_indices),
                ("id_test", id_test_indices),
            ]:
                part = df.loc[split_indices].copy()
                if part.empty:
                    raise RuntimeError(
                        f"OfficeHome split {scheme} produced an empty {new_split} split."
                    )
                part["split"] = new_split
                parts.append(part)

        target_indices = df.index[domain.eq(target_id)].to_numpy()
        rng.shuffle(target_indices)
        val_size = max(1, len(target_indices) // 2)
        for new_split, split_indices in [
            ("val", target_indices[:val_size]),
            ("test", target_indices[val_size:]),
        ]:
            part = df.loc[split_indices].copy()
            if part.empty:
                raise RuntimeError(
                    f"OfficeHome split {scheme} produced an empty {new_split} split."
                )
            part["split"] = new_split
            parts.append(part)
        return pd.concat(parts, axis=0, ignore_index=True)

    def get_input(self, idx):
        path = self._data_dir / self._input_array[idx]
        return Image.open(path).convert("RGB")

    def eval(self, y_pred, y_true, metadata):
        correct = (y_pred == y_true).float()
        return {"acc_avg": correct.mean().item()}, f"Average acc: {correct.mean().item():.3f}\n"
