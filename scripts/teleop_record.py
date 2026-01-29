import argparse
import sys
import os
import time
import h5py
import numpy as np
import torch
import json

from isaaclab.app import AppLauncher

# 1. Launch App (THIS MUST RUN FIRST)
parser = argparse.ArgumentParser(description="Teleop Recording for Imitation Learning")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--filename", type=str, default="stretch_cabinet_demo", help="Name of output file"
)
parser.add_argument(
    "--num_demos", type=int, default=10, help="Target number of successful demos"
)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input
import omni.appwindow
from isaaclab.envs import ManagerBasedRLEnv

# --- IMPORT YOUR CONFIG ---
# Ensure these paths match your actual directory structure
task_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(task_dir)
sys.path.append(os.path.join(task_dir, "config"))
from stretch_env_cfg import StretchEnvCfg


# =================================================================================
#  ROBOMIMIC DATA COLLECTOR (FIXED FOR NESTED DICTS)
# =================================================================================
class RobomimicDataCollector:
    """Saves data to HDF5 in Robomimic structure, handling nested Isaac Lab obs."""

    def __init__(self, env_name, directory_path, filename, num_demos):
        self.num_demos = num_demos

        # Create output directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        self.file_path = os.path.join(directory_path, f"{filename}.hdf5")
        self.f = h5py.File(self.file_path, "w")
        self.data_group = self.f.create_group("data")

        # Metadata
        self.data_group.attrs["total"] = 0
        self.data_group.attrs["env_args"] = json.dumps(
            {"env_name": env_name, "type": 1}
        )

        self.reset_buffer()
        print(f"[INFO] Data Collector initialized. Saving to: {self.file_path}")

    def reset_buffer(self):
        self.current_episode = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def reset(self):
        self.reset_buffer()

    def _to_numpy(self, value):
        """Recursive helper to convert tensors/dicts to numpy."""
        if isinstance(value, dict):
            return {k: self._to_numpy(v) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        else:
            return value

    def add(self, key, value):
        val_np = self._to_numpy(value)
        if key in self.current_episode:
            self.current_episode[key].append(val_np)

    def _save_dict_group(self, h5_parent, data_list, group_name):
        """Recursively saves a list of nested dictionaries into HDF5 groups."""
        # data_list is [frame1, frame2, ...] where frame1 is {"policy": {"joint_pos": ...}}

        # 1. Create the group (e.g. "obs")
        grp = h5_parent.create_group(group_name)

        # 2. Inspect structure of the first frame
        first_frame = data_list[0]

        if isinstance(first_frame, dict):
            for key in first_frame.keys():
                # Extract this key from all frames
                child_data_list = [frame[key] for frame in data_list]
                # Recurse (handles nested "policy" group)
                self._save_dict_group(grp, child_data_list, key)
        else:
            # It's a leaf node (array), stack and save
            data_stack = np.array(data_list).squeeze(
                axis=1
            )  # Remove batch dim if needed
            h5_parent.create_dataset(group_name, data=data_stack)

    def flush(self):
        demo_idx = self.data_group.attrs["total"]
        demo_group_name = f"demo_{demo_idx}"
        ep_grp = self.data_group.create_group(demo_group_name)

        # 1. Process Observations (Recursive)
        if len(self.current_episode["obs"]) > 0:
            # We strip the outer "obs" wrapper and just pass the list of obs dicts
            # But the helper expects to create the group name passed to it.
            # So we iterate keys manually for the top level.
            obs_grp = ep_grp.create_group("obs")
            first_obs = self.current_episode["obs"][0]

            # If obs is {"policy": ...}, we want data/demo_0/obs/policy/...
            for key in first_obs.keys():
                column = [x[key] for x in self.current_episode["obs"]]
                self._save_dict_group(obs_grp, column, key)

            # Do same for next_obs
            if len(self.current_episode["next_obs"]) > 0:
                next_obs_grp = ep_grp.create_group("next_obs")
                for key in first_obs.keys():
                    column = [x[key] for x in self.current_episode["next_obs"]]
                    self._save_dict_group(next_obs_grp, column, key)

        # 2. Process Actions, Rewards, Dones (Simple Arrays)
        for key in ["actions", "rewards", "dones"]:
            if self.current_episode[key]:
                data_stack = np.array(self.current_episode[key]).squeeze(axis=1)
                ep_grp.create_dataset(key, data=data_stack)

        # 3. Attributes
        ep_grp.attrs["num_samples"] = len(self.current_episode["actions"])
        ep_grp.attrs["model_file"] = "xml"  # Dummy attribute often checked by Mimic

        self.data_group.attrs["total"] += 1
        print(f"[INFO] Saved {demo_group_name} ({ep_grp.attrs['num_samples']} steps)")
        self.f.flush()
        self.reset_buffer()

    def is_stopped(self):
        return self.data_group.attrs["total"] >= self.num_demos

    def close(self):
        self.f.close()


# =================================================================================
#  INPUT CONTROLLER
# =================================================================================
class FullController:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()

    def is_pressed(self, key_name):
        # Handle special key names if needed
        if hasattr(carb.input.KeyboardInput, key_name):
            key_enum = getattr(carb.input.KeyboardInput, key_name)
            return self._input.get_keyboard_value(self._keyboard, key_enum) > 0.5
        return False


# =================================================================================
#  MAIN LOOP
# =================================================================================
def main():
    env_cfg = StretchEnvCfg()
    # Ensure env renders
    env_cfg.sim.render_interval = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    controller = FullController()

    # Log path
    log_dir = os.path.join(os.getcwd(), "datasets")  # Saving to datasets folder
    collector = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
    )

    print("-" * 40)
    print("[INFO] CONTROLS:")
    print("  Base: I, K (Fwd/Back) | J, L (Turn)")
    print("  Lift: C (Up), V (Down)")
    print("  Arm:  Z (Out), X (In)")
    print("  Wrist: N/M (Pitch), ,/. (Yaw), ;/' (Roll)")
    print("  Grip:  [ (Open), ] (Close)")
    print("  SAVE:  P  (Success)")
    print("  RESET: O  (Retry/Discard)")
    print("-" * 40)

    # Tuning
    BASE_SPEED = 1.0
    LIFT_SPEED = 0.5
    ARM_SPEED = 0.2
    WRIST_SPEED = 1.5

    # Gravity Compensation (Keeps arm from drooping when keys aren't pressed)
    # Adjust these if your arm drifts up or down!
    GRAVITY_LIFT_COMP = 0.0
    GRVITY_WRIST_PITCH_COMP = 0.0

    GRIPPER_CLOSE_POS = -0.1
    GRIPPER_OPEN_POS = 0.4
    gripper_goal = GRIPPER_CLOSE_POS

    obs, _ = env.reset()
    collector.reset()
    reset_needed = False

    while simulation_app.is_running():
        if collector.is_stopped():
            print(f"[SUCCESS] Collected {args_cli.num_demos} demos.")
            break

        with torch.inference_mode():
            # Check Reset
            if controller.is_pressed("O"):
                reset_needed = True

            # --- CONTROL LOGIC ---
            # 1. Base
            vel_fwd = 0.0
            vel_turn = 0.0
            if controller.is_pressed("I"):
                vel_fwd = BASE_SPEED
            if controller.is_pressed("K"):
                vel_fwd = -BASE_SPEED
            if controller.is_pressed("J"):
                vel_turn = BASE_SPEED
            if controller.is_pressed("L"):
                vel_turn = -BASE_SPEED

            # 2. Lift
            vel_lift = GRAVITY_LIFT_COMP
            if controller.is_pressed("C"):
                vel_lift = LIFT_SPEED
            if controller.is_pressed("V"):
                vel_lift = -LIFT_SPEED

            # 3. Arm
            vel_arm = 0.0
            if controller.is_pressed("Z"):
                vel_arm = ARM_SPEED
            if controller.is_pressed("X"):
                vel_arm = -ARM_SPEED

            # 4. Wrist
            vel_pitch = GRVITY_WRIST_PITCH_COMP
            if controller.is_pressed("N"):
                vel_pitch = WRIST_SPEED
            if controller.is_pressed("M"):
                vel_pitch = -WRIST_SPEED

            vel_yaw = 0.0
            if controller.is_pressed("COMMA"):
                vel_yaw = WRIST_SPEED
            if controller.is_pressed("PERIOD"):
                vel_yaw = -WRIST_SPEED

            vel_roll = 0.0
            if controller.is_pressed("SEMICOLON"):
                vel_roll = WRIST_SPEED
            if controller.is_pressed("APOSTROPHE"):
                vel_roll = -WRIST_SPEED

            # 5. Gripper
            if controller.is_pressed("LEFT_BRACKET"):
                gripper_goal = GRIPPER_OPEN_POS
            if controller.is_pressed("RIGHT_BRACKET"):
                gripper_goal = GRIPPER_CLOSE_POS

            actions = torch.zeros((env.num_envs, 12), device=env.device)

            actions[:, 0] = vel_turn
            actions[:, 1] = vel_fwd
            actions[:, 2] = vel_lift
            actions[:, 3:7] = vel_arm
            actions[:, 7] = vel_yaw
            actions[:, 8] = vel_pitch
            actions[:, 9] = vel_roll
            actions[:, 10] = gripper_goal
            actions[:, 11] = gripper_goal

            # Step
            next_obs, rew, terminated, truncated, info = env.step(actions)

            # --- RECORDING ---
            if reset_needed:
                print("[INFO] Resetting... Discarding episode.")
                obs, _ = env.reset()
                collector.reset()
                reset_needed = False
                continue

            collector.add("obs", obs)
            collector.add("actions", actions)
            collector.add("rewards", rew)
            collector.add("dones", terminated | truncated)
            collector.add("next_obs", next_obs)

            # Save on 'P'
            if controller.is_pressed("P"):
                collector.flush()
                obs, _ = env.reset()
                # Wait a few frames
                for _ in range(10):
                    env.render()
            else:
                obs = next_obs

    collector.close()
    env.close()


if __name__ == "__main__":
    main()
