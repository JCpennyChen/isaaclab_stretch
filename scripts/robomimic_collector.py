"""Lightweight Robomimic-compatible HDF5 data collector with episode discard support."""

import os
import json
import h5py
import torch
import numpy as np


class RobomimicDataCollector:
    """
    Writes HDF5 in robomimic format with support for:
    - Episode discard (reset_buffer without losing saved demos)
    - Train/valid split at write time
    """

    def __init__(self, env_name, directory_path, filename, num_demos, val_ratio=0.1):
        self.num_demos = num_demos
        self.val_ratio = val_ratio

        os.makedirs(directory_path, exist_ok=True)
        self.file_path = os.path.join(directory_path, f"{filename}.hdf5")
        self.f = h5py.File(self.file_path, "w")
        self.data_group = self.f.create_group("data")
        self.data_group.attrs["total"] = 0
        self.data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": env_name,
                "type": 1,
                "env_kwargs": {},
            }
        )

        self.train_demos = []
        self.valid_demos = []
        self.reset_buffer()
        print(f"[Collector] Initialized: {self.file_path}")

    def reset_buffer(self):
        """Discard current episode data without affecting saved demos."""
        self.buffer = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.flatten().detach().cpu().numpy()
        elif isinstance(value, dict):
            return {k: self._to_numpy(v) for k, v in value.items()}
        return value

    def add(self, key, value):
        self.buffer[key].append(self._to_numpy(value))

    def _save_nested(self, h5_parent, data_list, name):
        """Recursively save list of dicts or arrays into HDF5."""
        if isinstance(data_list[0], dict):
            grp = h5_parent.create_group(name)
            for key in data_list[0].keys():
                self._save_nested(grp, [frame[key] for frame in data_list], key)
        else:
            h5_parent.create_dataset(name, data=np.array(data_list))

    def flush(self):
        """Save current buffer as a demo and assign to train or valid split."""
        if len(self.buffer["actions"]) == 0:
            print("[Collector] Warning: empty buffer, skipping flush.")
            return

        demo_idx = self.data_group.attrs["total"]
        demo_name = f"demo_{demo_idx}"
        ep_grp = self.data_group.create_group(demo_name)

        # Save obs and next_obs (may be dicts)
        for obs_key in ["obs", "next_obs"]:
            if self.buffer[obs_key]:
                self._save_nested(ep_grp, self.buffer[obs_key], obs_key)

        # Save flat arrays
        for key in ["actions", "rewards", "dones"]:
            if self.buffer[key]:
                ep_grp.create_dataset(key, data=np.array(self.buffer[key]))

        ep_grp.attrs["num_samples"] = len(self.buffer["actions"])

        # Train/valid assignment
        is_last = (demo_idx + 1) >= self.num_demos
        if (np.random.rand() < self.val_ratio) or (
            is_last and len(self.valid_demos) == 0
        ):
            self.valid_demos.append(demo_name)
            split = "VALID"
        else:
            self.train_demos.append(demo_name)
            split = "TRAIN"

        self.data_group.attrs["total"] += 1
        print(
            f"[Collector] Saved {demo_name} -> {split} ({ep_grp.attrs['num_samples']} steps)"
        )

        self.f.flush()
        self.reset_buffer()

    def is_stopped(self):
        return self.data_group.attrs["total"] >= self.num_demos

    def close(self):
        if "mask" in self.f:
            del self.f["mask"]
        mask_grp = self.f.create_group("mask")
        mask_grp.create_dataset("train", data=np.array(self.train_demos, dtype="S"))
        mask_grp.create_dataset("valid", data=np.array(self.valid_demos, dtype="S"))
        print(
            f"[Collector] Closed. Train: {len(self.train_demos)}, Valid: {len(self.valid_demos)}"
        )
        self.f.close()
