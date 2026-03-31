"""
train_imitation.py

Trains BC-RNN policies for the Stretch cabinet task using robomimic.
Supports training reach phase, pull phase, or both sequentially.

Usage:
    python train_imitation.py                          # Train both phases
    python train_imitation.py --phase reach            # Train reach only
    python train_imitation.py --phase pull             # Train pull only
    python train_imitation.py --config my_config.json  # Train with a custom config
"""

import os
import sys
import json
import argparse
import time
import datetime
import torch

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train


# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CONFIG_DIR = os.path.join(
    PROJECT_ROOT, "source", "stretch", "stretch", "tasks",
    "manager_based", "stretch", "config", "robomimic",
)

DEFAULT_CONFIGS = {
    "reach": os.path.join(CONFIG_DIR, "bc_rnn_reach.json"),
    "pull": os.path.join(CONFIG_DIR, "bc_rnn_pull.json"),
}


def load_config(config_path):
    """Load a robomimic config from a JSON file."""
    print(f"[Config] Loading: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the JSON and create a robomimic config object
    with open(config_path, "r") as f:
        ext_cfg = json.load(f)

    config = config_factory(ext_cfg["algo_name"])
    config.update(ext_cfg)
    config.lock()

    # Validate that the dataset file exists
    data_path = config.train.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"  Run the recording script first to generate demo data."
        )

    return config


def train_phase(phase_name, config_path, device):
    """Train a single phase (reach or pull)."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {phase_name.upper()} PHASE")
    print(f"{'='*60}")

    config = load_config(config_path)

    # Print training summary
    print(f"  Dataset:     {config.train.data}")
    print(f"  Output:      {config.train.output_dir}")
    print(f"  Epochs:      {config.train.num_epochs}")
    print(f"  Batch size:  {config.train.batch_size}")
    print(f"  Seq length:  {config.train.seq_length}")
    print(f"  RNN hidden:  {config.algo.rnn.hidden_dim}")
    print(f"  RNN layers:  {config.algo.rnn.num_layers}")
    print(f"  Device:      {device}")
    print()

    start_time = time.time()
    train(config, device=device)
    elapsed = time.time() - start_time

    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    print(f"\n[DONE] {phase_name.upper()} training completed in {elapsed_str}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Train BC-RNN imitation learning policies for Stretch cabinet task"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["reach", "pull"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (default: auto-detect)",
    )
    args = parser.parse_args()

    # Device setup
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if args.device is not None:
        device = torch.device(args.device)
    print(f"[Setup] Using device: {device}")

    total_start = time.time()

    # Custom config mode: train a single config and exit
    if args.config is not None:
        phase_name = os.path.splitext(os.path.basename(args.config))[0]
        train_phase(phase_name, args.config, device)
        return

    # Standard mode: train reach, pull, or both
    phases_to_train = []
    if args.phase in ("reach"):
        phases_to_train.append(("reach", DEFAULT_CONFIGS["reach"]))
    if args.phase in ("pull"):
        phases_to_train.append(("pull", DEFAULT_CONFIGS["pull"]))

    # Validate all configs exist before starting
    for phase_name, config_path in phases_to_train:
        if not os.path.exists(config_path):
            print(f"[ERROR] Config not found: {config_path}")
            sys.exit(1)

    # Train each phase
    timings = {}
    for phase_name, config_path in phases_to_train:
        elapsed = train_phase(phase_name, config_path, device)
        timings[phase_name] = elapsed

    # Final summary
    total_elapsed = time.time() - total_start
    total_str = str(datetime.timedelta(seconds=int(total_elapsed)))

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for phase_name, elapsed in timings.items():
        phase_str = str(datetime.timedelta(seconds=int(elapsed)))
        print(f"  {phase_name:>6}: {phase_str}")
    print(f"  {'Total':>6}: {total_str}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
