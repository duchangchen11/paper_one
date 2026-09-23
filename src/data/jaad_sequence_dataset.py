"""PyTorch dataset for the processed JAAD sequence files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class JAADSequenceDataset(Dataset):
    """Load one split of the processed JAAD arrays into memory."""

    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=False)
        self.target_obs = torch.from_numpy(data["target_obs"].astype(np.float32))
        self.target_abs_obs = torch.from_numpy(data["target_abs_obs"].astype(np.float32))
        self.future_gt = torch.from_numpy(data["future_gt"].astype(np.float32))
        self.neighbor_obs = torch.from_numpy(data["neighbor_obs"].astype(np.float32))
        self.neighbor_mask = torch.from_numpy(data["neighbor_mask"].astype(np.float32))
        self.neighbor_visible_mask = torch.from_numpy(
            data["neighbor_visible_mask"].astype(np.float32)
        )
        self.intent_label = torch.from_numpy(data["intent_label"].astype(np.float32))
        if "scene_feat" in data.files:
            self.scene_feat = torch.from_numpy(data["scene_feat"].astype(np.float32))
        else:
            self.scene_feat = torch.zeros((self.target_obs.shape[0], 0), dtype=torch.float32)

    def __len__(self) -> int:
        return self.target_obs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target_obs": self.target_obs[index],
            "target_abs_obs": self.target_abs_obs[index],
            "future_gt": self.future_gt[index],
            "neighbor_obs": self.neighbor_obs[index],
            "neighbor_mask": self.neighbor_mask[index],
            "neighbor_visible_mask": self.neighbor_visible_mask[index],
            "scene_feat": self.scene_feat[index],
            "intent_label": self.intent_label[index],
        }
