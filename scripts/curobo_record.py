import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import h5py
import json
import torch
import argparse
import numpy as np

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

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

# ==========================================
# ENVIRONMENT & UTILITY IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if target_config_dir not in sys.path:
    sys.path.append(target_config_dir)

from stretch_env_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import combine_frame_transforms
from isaacsim.core.prims import XFormPrim

# ==========================================
# CUROBO IMPORTS
# ==========================================
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType


# =================================================================================
#  ROBOMIMIC DATA COLLECTOR
# =================================================================================
class RobomimicDataCollector:
    """Saves data to HDF5 in Robomimic structure, handling nested Isaac Lab obs."""

    def __init__(self, env_name, directory_path, filename, num_demos, val_ratio=0.1):
        self.num_demos = num_demos
        self.val_ratio = val_ratio

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        self.file_path = os.path.join(directory_path, f"{filename}.hdf5")
        self.f = h5py.File(self.file_path, "w")
        self.data_group = self.f.create_group("data")
        env_args = {
            "env_name": env_name,
            "type": 1,
            "env_kwargs": {},
        }
        self.data_group.attrs["total"] = 0
        self.data_group.attrs["env_args"] = json.dumps(env_args)
        self.train_demos = []
        self.valid_demos = []

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
        first_frame = data_list[0]

        if isinstance(first_frame, dict):
            grp = h5_parent.create_group(group_name)
            for key in first_frame.keys():
                child_data_list = [frame[key] for frame in data_list]
                self._save_dict_group(grp, child_data_list, key)
        else:
            data_stack = np.array(data_list).squeeze(axis=1)
            h5_parent.create_dataset(group_name, data=data_stack)

    def flush(self):
        demo_idx = self.data_group.attrs["total"]
        demo_group_name = f"demo_{demo_idx}"
        ep_grp = self.data_group.create_group(demo_group_name)

        if len(self.current_episode["obs"]) > 0:
            obs_grp = ep_grp.create_group("obs")
            first_obs = self.current_episode["obs"][0]
            if isinstance(first_obs, dict):
                for key in first_obs.keys():
                    column = [x[key] for x in self.current_episode["obs"]]
                    self._save_dict_group(obs_grp, column, key)
            else:
                print("[Warning] Obs is not a dict, saving as raw 'obs'")
                ep_grp.create_dataset("obs", data=np.array(self.current_episode["obs"]))

            if len(self.current_episode["next_obs"]) > 0:
                next_obs_grp = ep_grp.create_group("next_obs")
                if isinstance(first_obs, dict):
                    for key in first_obs.keys():
                        column = [x[key] for x in self.current_episode["next_obs"]]
                        self._save_dict_group(next_obs_grp, column, key)

        for key in ["actions", "rewards", "dones"]:
            if self.current_episode[key]:
                data_stack = np.array(self.current_episode[key]).squeeze(axis=1)
                ep_grp.create_dataset(key, data=data_stack)

        ep_grp.attrs["num_samples"] = len(self.current_episode["actions"])
        ep_grp.attrs["model_file"] = "xml"

        if np.random.rand() < self.val_ratio:
            self.valid_demos.append(demo_group_name)
            split_name = "VALID"
        else:
            self.train_demos.append(demo_group_name)
            split_name = "TRAIN"

        self.data_group.attrs["total"] += 1
        print(
            f"[INFO] Saved {demo_group_name} to {split_name} ({ep_grp.attrs['num_samples']} steps)"
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
            f"[INFO] Closing file. Final Split -> Train: {len(self.train_demos)}, Valid: {len(self.valid_demos)}"
        )
        self.f.close()


def main():
    env_cfg = StretchEnvCfg()
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)
    env_cfg.episode_length_s = 10000.0
    print("[Fix] forcing High Stiffness (4000.0) on all joints...")
    strong_drive = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=4000.0,
        damping=100.0,
        effort_limit=20000.0,
        velocity_limit=100.0,
    )
    env_cfg.scene.robot.actuators = {"god_mode_drive": strong_drive}
    env_cfg.actions.base = None
    env_cfg.actions.lift_velocity = None
    env_cfg.actions.arm_velocity = None
    env_cfg.actions.wrist_velocity = None
    env_cfg.actions.gripper = None
    env_cfg.actions.joint_pos_direct = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )

    log_dir = os.path.join(os.getcwd(), "datasets")
    collector = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
    )

    collector.reset()
    print("[IsaacLab] Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    curobo_config_path = "/home/johnchen/SharedSSD/JohnChen/stretch/assets/configs/robot_configs/stretch_joint.yml"
    if not os.path.exists(curobo_config_path):
        raise FileNotFoundError(f"Could not find CuRobo config at {curobo_config_path}")

    robot_cfg = load_yaml(curobo_config_path)["robot_cfg"]

    dummy_world = WorldConfig(
        cuboid=[
            Cuboid(
                "startup_dummy", pose=[0, 0, -10.0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0]
            )
        ]
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        dummy_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        optimize_dt=True,
        trajopt_tsteps=32,
    )

    motion_gen = MotionGen(motion_gen_config)
    print("[CuRobo] Warming up...")
    motion_gen.warmup(enable_graph=True)

    usd_help = UsdHelper()
    usd_help.load_stage(env.sim.stage)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=10,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    cabinet_prim_path = "/World/envs/env_0/Cabinet"
    cabinet_view = XFormPrim(cabinet_prim_path, name="cabinet")

    default_cabinet_pos, default_cabinet_rot = cabinet_view.get_world_poses()
    default_cabinet_pos = default_cabinet_pos[0].clone()

    def randomize_cabinet():
        noise_range = 0.5

        rand_x = (torch.rand(1, device=env.device) * 2 - 1) * noise_range
        rand_y = (torch.rand(1, device=env.device) * 2 - 1) * noise_range

        new_pos = default_cabinet_pos.clone()
        new_pos[0] += rand_x[0]
        new_pos[1] += rand_y[0]

        cabinet_view.set_world_poses(
            positions=new_pos.unsqueeze(0), orientations=default_cabinet_rot
        )

        cabinet_articulation = env.scene["cabinet"]
        root_state = cabinet_articulation.data.default_root_state.clone()
        root_state[:, :3] = new_pos

        cabinet_articulation.write_root_pose_to_sim(root_state[:, :7])
        cabinet_articulation.write_root_velocity_to_sim(root_state[:, 7:])

        base_env_path = "/World/envs/env_0"
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=[f"{base_env_path}/Cabinet", f"{base_env_path}/ObstacleCube"],
            ignore_substring=[f"{base_env_path}/Robot"],
        ).get_collision_check_world()
        motion_gen.update_world(obstacles)

        print(f"[Random] Cabinet moved to: {new_pos.cpu().numpy()}")

    cabinet_entity = env.scene["cabinet"]
    handle_body_idx = cabinet_entity.get_body_index_from_name("drawer_handle_body")

    randomize_cabinet()

    trajectory = None
    traj_idx = 0
    step_count = 0
    target_pose = None

    target_frame_path = (
        "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"
    )
    target_frame_view = XFormPrim(target_frame_path, name="target_frame")
    phase_one_done = False
    phase_two_done = False
    phase_three_done = False

    transition_timer = 0
    gripper_timer = 0
    hold_joints = None

    success_hold_timer = 0

    initial_handle_pos = None
    ik_fail_count = 0

    print(">>> Starting Simulation Loop...")
    while simulation_app.is_running():
        if collector.is_stopped():
            print(f"[SUCCESS] Collected {args_cli.num_demos} demos. Stopping.")
            break

        if target_pose is None:
            curr_pos, curr_quat = target_frame_view.get_world_poses()
            physics_handle_pos = cabinet_entity.data.body_pos_w[
                0, handle_body_idx
            ].clone()
            if initial_handle_pos is None:
                initial_handle_pos = physics_handle_pos.clone()

            # PHASE 1: SAFE SPOT
            if not phase_one_done:
                print("[Logic] Phase 1: Planning to Safe Spot (-0.08)...")
                front_offset = torch.tensor([[-0.08, 0.03, 0.0]], device=env.device)

            # PHASE 2: INSERT
            elif not phase_two_done:
                print("[Logic] Phase 2: Planning Insert (0.05)...")
                front_offset = torch.tensor([[0.1, 0.03, 0.0]], device=env.device)

            # PHASE 3: PULL BACK
            else:
                print("[Logic] Phase 3: Planning Pull (-0.45)...")
                front_offset = torch.tensor([[-0.45, 0.03, 0.0]], device=env.device)

            target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)
            target_pos_w, _ = combine_frame_transforms(
                curr_pos[0:1], target_quat_w, front_offset
            )
            target_pose = Pose(position=target_pos_w, quaternion=target_quat_w)

        # --- OBSTACLE UPDATE ---
        if step_count % 60 == 0 and step_count > 0:
            base_env_path = "/World/envs/env_0"
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=[
                    f"{base_env_path}/Cabinet",
                    f"{base_env_path}/ObstacleCube",
                ],
                ignore_substring=[f"{base_env_path}/Robot"],
            ).get_collision_check_world()
            motion_gen.update_world(obstacles)

        robot_entity = env.scene["robot"]
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        is_static = robot_velocity < 0.5

        # CHANGE THIS LINE: "step_count > 50" -> "step_count > 5"
        if trajectory is None and step_count > 5 and is_static:
            print("[CuRobo] Planning path...")
            cu_js = JointState(
                position=robot_entity.data.joint_pos[0].unsqueeze(0),
                velocity=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                acceleration=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                joint_names=robot_entity.joint_names,
            ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

            result = motion_gen.plan_single(cu_js, target_pose, plan_config)

            if result.success.item():
                print(
                    f"[CuRobo] Success! Steps: {result.optimized_plan.position.shape[1]}"
                )
                trajectory = result.get_interpolated_plan()
                trajectory = motion_gen.get_full_js(trajectory)
                traj_idx = 0
                ik_fail_count = 0
            else:
                ik_fail_count += 1
                if ik_fail_count > 5:
                    print(
                        "[Logic] Too many IK failures (Position Unreachable). Resetting..."
                    )
                    obs, _ = env.reset()
                    randomize_cabinet()
                    trajectory = None
                    target_pose = None
                    traj_idx = 0
                    phase_one_done = False
                    phase_two_done = False
                    phase_three_done = False
                    hold_joints = None
                    gripper_timer = 0
                    success_hold_timer = 0
                    transition_timer = 0
                    initial_handle_pos = None
                    ik_fail_count = 0
                    motion_gen.reset()

        actions = robot_entity.data.joint_pos.clone()

        if hold_joints is not None:
            actions = hold_joints.clone()
        else:
            actions = robot_entity.data.joint_pos.clone()

        if trajectory is not None:
            if traj_idx >= len(trajectory.position):
                if not phase_one_done:
                    transition_timer += 1
                    target_state = trajectory[-1]
                    if transition_timer > 30:
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        phase_one_done = True
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        transition_timer = 0

                elif not phase_two_done:
                    target_state = trajectory[-1]
                    gripper_timer += 1

                    if gripper_timer > 60:
                        print("--> Gripper Locked! Switching to Phase 3 (Pull)...")
                        phase_two_done = True
                        hold_joints = actions.clone()
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        gripper_timer = 0

                else:
                    target_state = trajectory[-1]
                    phase_three_done = True

            else:
                target_state = trajectory[traj_idx]
                traj_idx += 1

            if trajectory is not None:
                flat_pos = target_state.position.view(-1)
                target_pos_dict = {
                    name: flat_pos[i] for i, name in enumerate(trajectory.joint_names)
                }
                for i, name in enumerate(env.scene["robot"].joint_names):
                    if name in target_pos_dict:
                        actions[0, i] = target_pos_dict[name]

        gripper_idx = -1
        if phase_three_done:
            actions[0, gripper_idx] = 0.2
        elif not phase_two_done:
            if (
                phase_one_done
                and trajectory is not None
                and traj_idx >= len(trajectory.position)
            ):
                actions[0, gripper_idx] = -0.5
                if gripper_timer % 30 == 0:
                    print("Clamping Gripper...")
            else:
                actions[0, gripper_idx] = 0.2

        else:
            actions[0, gripper_idx] = -0.5

        next_obs, rew, terminated, truncated, extras = env.step(actions)
        collector.add("obs", obs)
        collector.add("actions", actions)
        collector.add("rewards", rew)
        collector.add("dones", terminated | truncated)
        collector.add("next_obs", next_obs)

        obs = next_obs
        step_count += 1
        if phase_three_done:
            success_hold_timer += 1
            if success_hold_timer > 30:
                final_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
                displacement = torch.norm(final_pos - initial_handle_pos)
                if displacement > 0.07:
                    print(
                        f"[Record] SUCCESS! Drawer opened {displacement:.2f}m. Saving..."
                    )
                    collector.flush()
                else:
                    print(
                        f"[Record] FAIL. Drawer only moved {displacement:.2f}m. Discarding."
                    )
                    collector.reset_buffer()

                obs, _ = env.reset()
                randomize_cabinet()
                trajectory = None
                target_pose = None
                traj_idx = 0
                phase_one_done = False
                phase_two_done = False
                phase_three_done = False
                hold_joints = None
                gripper_timer = 0
                success_hold_timer = 0
                transition_timer = 0
                initial_handle_pos = None
                motion_gen.reset()

    env.close()
    collector.close()


if __name__ == "__main__":
    main()
