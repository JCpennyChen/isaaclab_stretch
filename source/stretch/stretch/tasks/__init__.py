# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

import gymnasium as gym
import os
from isaaclab_tasks.utils import import_packages

# 1. Import your Environment Configuration
# Make sure this path matches where your stretch_env_cfg.py actually lives
from .manager_based.stretch.stretch_env_cfg import StretchEnvCfg

# 2. Define the path to your Training Config (bc.yaml)
# This assumes you created the file at: .../tasks/manager_based/stretch/config/robomimic/bc.json
current_dir = os.path.dirname(os.path.abspath(__file__))
bc_config_path = os.path.join(
    current_dir, "manager_based/stretch/config/robomimic/bc.json"
)

# 3. Register the Task
gym.register(
    id="Template-Stretch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": StretchEnvCfg,
        "robomimic_bc_cfg_entry_point": bc_config_path,  # <--- This fixes the KeyError
    },
    disable_env_checker=True,
)

# 4. (Optional) Keep the auto-importer if you have other tasks
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
