"""
student_policy_train.py

Train a student policy (CNN + MLP) on distillation data collected by
student_policy_record.py.

Input:  proprioception (26D: arm(8)+base(3)+vel(11)+gripper(2)+head(2)) + RGB image (480x640x3)
Output: actions (13D: arm(8) + base(3) + gripper(2), no head deltas)

Usage:
    python scripts/bc_rnn/student_policy_train.py \
        --reach_data datasets/distillation_reach.hdf5 \
        --pull_data datasets/distillation_pull.hdf5 \
        --epochs 100

    # Reach only
    python scripts/bc_rnn/student_policy_train.py \
        --reach_data datasets/distillation_reach.hdf5 \
        --epochs 100
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ==========================================
# Dataset
# ==========================================
class DistillationDataset(Dataset):
    """Loads distillation HDF5 data. Each sample is one timestep."""

    def __init__(self, hdf5_path, split="train", image_size=(120, 160)):
        self.image_size = image_size  # (H, W) to resize to
        self.proprio_list = []
        self.image_list = []
        self.action_list = []

        with h5py.File(hdf5_path, "r") as f:
            # Get demo names for this split
            mask = f["mask"]
            if split not in mask or len(mask[split]) == 0:
                print(f"[Dataset] {hdf5_path} ({split}): 0 demos")
                self.proprio = np.zeros((0, 26), dtype=np.float32)
                self.images = np.zeros((0,), dtype=np.uint8)
                self.actions = np.zeros((0, 13), dtype=np.float32)
                return
            demo_names = [n.decode() for n in mask[split][:]]
            print(f"[Dataset] {hdf5_path} ({split}): {len(demo_names)} demos")

            for demo_name in demo_names:
                ep = f["data"][demo_name]
                # Load proprio components and concatenate
                arm = ep["obs/proprio/arm_joint_pos"][:]   # (T, 8)
                base = ep["obs/proprio/base_pos"][:]        # (T, 3)
                vel = ep["obs/proprio/joint_vel"][:]        # (T, 11)
                grip = ep["obs/proprio/gripper_state"][:]   # (T, 2)
                head = ep["obs/proprio/head_pos"][:]        # (T, 2)
                proprio = np.concatenate(
                    [arm, base, vel, grip, head], axis=-1
                )  # (T, 26)

                # Load images
                images = ep["obs/image"][:]  # (T, H, W, 3) uint8

                # Load actions
                actions = ep["actions"][:]  # (T, 13)

                self.proprio_list.append(proprio)
                self.image_list.append(images)
                self.action_list.append(actions)

        self.proprio = np.concatenate(self.proprio_list, axis=0).astype(np.float32)
        self.images = np.concatenate(self.image_list, axis=0)
        self.actions = np.concatenate(self.action_list, axis=0).astype(np.float32)

        print(f"  Total steps: {len(self.proprio)}")
        print(f"  Proprio shape: {self.proprio.shape}")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Action shape: {self.actions.shape}")

    def __len__(self):
        return len(self.proprio)

    def __getitem__(self, idx):
        # Proprio
        proprio = torch.tensor(self.proprio[idx])

        # Image: flattened uint8 (921600,) -> float (3, h, w) resized
        img = self.images[idx].reshape(480, 640, 3)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # (3, 480, 640)
        # Resize via interpolate
        img = nn.functional.interpolate(
            img.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # (3, 120, 160)

        # Action target
        action = torch.tensor(self.actions[idx])

        return proprio, img, action


# ==========================================
# Model
# ==========================================
class StudentPolicy(nn.Module):
    """Simple CNN image encoder + MLP policy."""

    def __init__(self, proprio_dim=26, action_dim=13, image_size=(120, 160)):
        super().__init__()

        # CNN encoder for image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * 4 * 4  # 1024

        # MLP: proprio + image features -> actions
        mlp_in = cnn_out_dim + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, proprio, image):
        img_feat = self.cnn(image)
        x = torch.cat([img_feat, proprio], dim=-1)
        return self.mlp(x)


# ==========================================
# Training
# ==========================================
def train_one_phase(name, hdf5_path, args, device):
    """Train a student policy on one phase (reach or pull)."""
    print(f"\n{'='*60}")
    print(f"  Training {name.upper()} policy")
    print(f"{'='*60}")

    image_size = (args.image_h, args.image_w)

    train_dataset = DistillationDataset(hdf5_path, split="train", image_size=image_size)
    valid_dataset = DistillationDataset(hdf5_path, split="valid", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = StudentPolicy(proprio_dim=26, action_dim=13, image_size=image_size).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Checkpoint dir
    ckpt_dir = os.path.join(args.output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for proprio, image, action in train_loader:
            proprio = proprio.to(device)
            image = image.to(device)
            action = action.to(device)

            pred = model(proprio, image)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * proprio.size(0)

        train_loss /= len(train_dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for proprio, image, action in valid_loader:
                proprio = proprio.to(device)
                image = image.to(device)
                action = action.to(device)

                pred = model(proprio, image)
                loss = loss_fn(pred, action)
                val_loss += loss.item() * proprio.size(0)
        val_loss /= len(valid_dataset)

        scheduler.step()

        # --- Logging & best checkpoint ---
        lr = optimizer.param_groups[0]["lr"]
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                os.path.join(ckpt_dir, "best.pth"),
            )
            marker = " *"

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4d}/{args.epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"lr {lr:.2e}{marker}"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                os.path.join(ckpt_dir, f"epoch_{epoch}.pth"),
            )

    # Save final
    torch.save(
        {"epoch": args.epochs, "model": model.state_dict(), "val_loss": val_loss},
        os.path.join(ckpt_dir, "last.pth"),
    )

    best_path = os.path.join(ckpt_dir, "best.pth")
    print(f"\n  Best epoch:    {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best ckpt:     {best_path}")
    print(f"  All ckpts:     {ckpt_dir}/")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train student distillation policy")
    parser.add_argument(
        "--reach_data", type=str, required=True, help="Path to reach HDF5 dataset"
    )
    parser.add_argument(
        "--pull_data", type=str, default=None, help="Path to pull HDF5 dataset"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--image_h", type=int, default=120, help="Resized image height")
    parser.add_argument("--image_w", type=int, default=160, help="Resized image width")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/johnchen/SharedSSD/JohnChen/stretch/checkpoints/student",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Train reach policy
    train_one_phase("reach", args.reach_data, args, device)

    # Train pull policy (if data provided)
    if args.pull_data is not None:
        train_one_phase("pull", args.pull_data, args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
