import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import torch
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stretch Cabinet Demo Recording")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--filename",
    type=str,
    default="stretch_cabinet_demo",
    help="Base name of output files",
)
parser.add_argument(
    "--num_demos", type=int, default=10, help="Target number of successful demos"
)
parser.add_argument("--ratio", type=float, default=0.1, help="Validation split ratio")
parser.add_argument(
    "--noise_range",
    type=float,
    default=0.2,
    help="Cabinet position randomization range (meters)",
)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(target_config_dir)


from bc_rnn_stretch_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaacsim.core.prims import XFormPrim
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_inv,
    quat_apply,
    quat_mul,
)

# ==========================================
# CUROBO IMPORTS
# ==========================================
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CUROBO_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "assets", "robot_configs", "stretch_fake_joint.yml"
)
ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

PLANNER_CONFIG = {
    "max_attempts": 10,
    "time_dilation_factor": 0.5,
    "enable_graph_attempt": 2,
}

PHASE_OFFSETS = {
    "safe_spot": [0.05, 0.0, 0.03],
    "insert": [-0.05, 0.0, 0.03],
    "pull": [0.45, 0.0, 0.03],
}

LINEAR_PLAN_CONFIG = MotionGenPlanConfig(
    enable_graph=False,
    enable_finetune_trajopt=True,
    max_attempts=1,
    time_dilation_factor=1.0,
)

TRANSITION_WAIT_STEPS = 30
GRIPPER_LOCK_STEPS = 30
GRIPPER_OPEN_POS = 0.1
GRIPPER_CLOSE_POS = -0.1
DRAWER_OPEN_THRESHOLD = 0.35
MAX_PLAN_FAILURES = 5
SUCCESS_HOLD_STEPS = 30
HANDLE_DISPLACEMENT_THRESHOLD = 0.05


sys.path.append(os.path.join(SCRIPT_DIR, "..", "tools"))
from robomimic_collector import RobomimicDataCollector


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def compute_phase_targets(target_frame_view, cabinet_view, robot_entity, device):
    """Compute all three phase target poses in the robot's base frame."""
    curr_pos, _ = target_frame_view.get_world_poses()
    _, cabinet_rot = cabinet_view.get_world_poses()
    robot_pos_w = robot_entity.data.root_state_w[0:1, :3]
    robot_quat_w = robot_entity.data.root_state_w[0:1, 3:7]
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=device)
    target_quat_r = quat_mul(quat_inv(robot_quat_w), target_quat_w)

    targets = {}
    for phase_name, offset in PHASE_OFFSETS.items():
        offset_t = torch.tensor([offset], device=device)
        target_pos_w, _ = combine_frame_transforms(
            curr_pos[0:1], cabinet_rot, offset_t, identity_quat
        )
        target_pos_r = quat_apply(quat_inv(robot_quat_w), target_pos_w - robot_pos_w)
        targets[phase_name] = Pose(position=target_pos_r, quaternion=target_quat_r)
    return targets


def randomize_cabinet(
    cabinet_view, cabinet_articulation, default_pos, default_rot, device, noise_range
):
    """Randomize cabinet XY position within noise_range."""
    new_pos = default_pos.clone()
    new_pos[0] += (torch.rand(1, device=device) * 2 - 1).item() * noise_range
    new_pos[1] += (torch.rand(1, device=device) * 2 - 1).item() * noise_range

    cabinet_view.set_world_poses(
        positions=new_pos.unsqueeze(0), orientations=default_rot
    )
    root_state = cabinet_articulation.data.default_root_state.clone()
    root_state[:, :3] = new_pos
    cabinet_articulation.write_root_pose_to_sim(root_state[:, :7])
    cabinet_articulation.write_root_velocity_to_sim(root_state[:, 7:])


def reset_episode_state():
    """Return a fresh set of episode state variables."""
    return {
        "trajectory": None,
        "traj_idx": 0,
        "step_count": 0,
        "phase_one_done": False,
        "phase_two_done": False,
        "phase_three_done": False,
        "transition_timer": 0,
        "gripper_timer": 0,
        "success_hold_timer": 0,
        "plan_fail_count": 0,
        "current_phase_name": "safe_spot",
        "phase3_start_step": None,
    }


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    env_cfg = StretchEnvCfg()

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    # ==========================================
    # Data Collectors
    # ==========================================
    log_dir = os.path.join(os.getcwd(), "datasets")

    collector_reach = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=log_dir,
        filename=args_cli.filename + "_reach",
        num_demos=args_cli.num_demos,
        val_ratio=args_cli.ratio,
    )
    collector_pull = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=log_dir,
        filename=args_cli.filename + "_pull",
        num_demos=args_cli.num_demos,
        val_ratio=args_cli.ratio,
    )

    # ==========================================
    # CuRobo Setup
    # ==========================================
    robot_cfg = load_yaml(CUROBO_CONFIG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSET_ROOT

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
        optimize_dt=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        trajopt_tsteps=40,
        interpolation_dt=0.0167,
    )
    motion_gen = MotionGen(motion_gen_config)

    if (
        hasattr(motion_gen, "self_collision_checker")
        and motion_gen.self_collision_checker is not None
    ):
        motion_gen.self_collision_checker.min_dist = 0.005

    print("[CuRobo] Warming up...")
    motion_gen.warmup(enable_graph=True)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_finetune_trajopt=True,
        **PLANNER_CONFIG,
    )

    # ==========================================
    # Scene References & Joint Index Resolution
    # ==========================================
    robot_entity = env.scene["robot"]
    cabinet_entity = env.scene["cabinet"]

    arm_joint_names = [
        "joint_lift",
        "joint_arm_l0",
        "joint_arm_l1",
        "joint_arm_l2",
        "joint_arm_l3",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    base_joint_names = ["joint_x", "joint_y", "joint_rot_z"]
    gripper_joint_names = ["joint_gripper_finger_left", "joint_gripper_finger_right"]

    arm_joint_ids_isaac = robot_entity.find_joints(arm_joint_names)[0]
    base_joint_ids_isaac = robot_entity.find_joints(base_joint_names)[0]
    gripper_joint_ids_isaac = robot_entity.find_joints(gripper_joint_names)[0]

    curobo_names = motion_gen.kinematics.joint_names
    arm_ids_curobo = [curobo_names.index(n) for n in arm_joint_names]
    base_ids_curobo = [curobo_names.index(n) for n in base_joint_names]

    handle_body_idx = cabinet_entity.data.body_names.index("drawer_handle_top")

    # ==========================================
    # Cabinet Setup
    # ==========================================
    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    cabinet_view = XFormPrim("/World/envs/env_0/Cabinet", name="cabinet")
    default_cabinet_pos, default_cabinet_rot = cabinet_view.get_world_poses()
    default_cabinet_pos = default_cabinet_pos[0].clone()

    # Initial randomization
    randomize_cabinet(
        cabinet_view,
        cabinet_entity,
        default_cabinet_pos,
        default_cabinet_rot,
        env.device,
        args_cli.noise_range,
    )

    # Compute initial targets and handle position
    phase_targets = compute_phase_targets(
        target_frame_view, cabinet_view, robot_entity, env.device
    )
    initial_handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx].clone()

    # ==========================================
    # Episode State
    # ==========================================
    ep = reset_episode_state()
    target_pose = phase_targets[ep["current_phase_name"]]

    demos_collected = 0
    demos_failed = 0
    sim_start_time = time.time()

    print(">>> Starting Recording Loop...")
    while simulation_app.is_running():
        if collector_reach.is_stopped() or collector_pull.is_stopped():
            print(f"\n[DONE] Collected {args_cli.num_demos} demos.")
            break

        # ==========================================
        # PLANNER TRIGGER
        # ==========================================
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        is_static = robot_velocity < 2.0

        # Skip is_static for Phase 3: gripper oscillation keeps velocity high
        ready_to_plan = is_static or ep["phase_two_done"]

        if (
            ep["trajectory"] is None
            and ep["step_count"] > 5
            and ready_to_plan
            and not ep["phase_three_done"]
        ):
            if ep["phase_two_done"]:
                current_phase = "3"
            elif ep["phase_one_done"]:
                current_phase = "2"
            else:
                current_phase = "1"

            print(f"[CuRobo] Planning Phase {current_phase}...")

            cu_js = JointState(
                position=robot_entity.data.joint_pos[0].unsqueeze(0),
                velocity=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                acceleration=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                joint_names=robot_entity.joint_names,
            ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

            active_config = (
                LINEAR_PLAN_CONFIG if current_phase in ["2", "3"] else plan_config
            )
            result = motion_gen.plan_single(cu_js, target_pose, active_config)

            if result.success.item():
                traj_len = result.optimized_plan.position.shape[1]
                print(f"  Phase {current_phase} PLAN SUCCESS! ({traj_len} steps)")

                ep["trajectory"] = result.get_interpolated_plan()
                ep["trajectory"] = motion_gen.get_full_js(ep["trajectory"])
                ep["traj_idx"] = 0
                ep["plan_fail_count"] = 0
            else:
                print(f"  Phase {current_phase} PLAN FAILED: {result.status}")
                ep["plan_fail_count"] += 1

                if ep["plan_fail_count"] >= MAX_PLAN_FAILURES:
                    print(
                        f"  [RESET] {MAX_PLAN_FAILURES} failures. Discarding episode."
                    )
                    collector_reach.reset_buffer()
                    collector_pull.reset_buffer()
                    demos_failed += 1

                    obs, _ = env.reset()
                    randomize_cabinet(
                        cabinet_view,
                        cabinet_entity,
                        default_cabinet_pos,
                        default_cabinet_rot,
                        env.device,
                        args_cli.noise_range,
                    )
                    phase_targets = compute_phase_targets(
                        target_frame_view, cabinet_view, robot_entity, env.device
                    )
                    initial_handle_pos = cabinet_entity.data.body_pos_w[
                        0, handle_body_idx
                    ].clone()
                    ep = reset_episode_state()
                    target_pose = phase_targets[ep["current_phase_name"]]
                    motion_gen.reset()
                    continue

        # ==========================================
        # Execution Logic (Joint Position)
        # ==========================================
        if ep["trajectory"] is not None:
            if ep["traj_idx"] >= len(ep["trajectory"].position):
                # End of trajectory
                target_state = ep["trajectory"][-1]

                if not ep["phase_one_done"]:
                    ep["transition_timer"] += 1
                    if ep["transition_timer"] > TRANSITION_WAIT_STEPS:
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        ep["phase_one_done"] = True
                        ep["trajectory"] = None
                        ep["current_phase_name"] = "insert"
                        target_pose = phase_targets[ep["current_phase_name"]]
                        ep["traj_idx"] = 0
                        ep["transition_timer"] = 0

                elif not ep["phase_two_done"]:
                    ep["gripper_timer"] += 1
                    if ep["gripper_timer"] > GRIPPER_LOCK_STEPS:
                        print("--> Gripper Locked! Switching to Phase 3...")
                        ep["phase_two_done"] = True
                        ep["trajectory"] = None
                        ep["current_phase_name"] = "pull"
                        target_pose = phase_targets[ep["current_phase_name"]]
                        ep["traj_idx"] = 0
                        ep["gripper_timer"] = 0
                        ep["phase3_start_step"] = ep["step_count"]

            else:
                # Mid-trajectory
                target_state = ep["trajectory"][ep["traj_idx"]]
                joint_error = torch.norm(
                    target_state.position[arm_ids_curobo]
                    - robot_entity.data.joint_pos[0, arm_joint_ids_isaac]
                ).item()
                # Looser tolerance during pull phase (drawer resists)
                threshold = 0.15 if ep["phase_two_done"] else 0.05
                if joint_error < threshold:
                    ep["traj_idx"] += 1

            # --- Phase 3 drawer check (runs every step, independent of traj progress) ---
            if ep["phase_two_done"] and not ep["phase_three_done"]:
                drawer_pos = cabinet_entity.data.joint_pos[0]
                if drawer_pos.max().item() > DRAWER_OPEN_THRESHOLD:
                    print(
                        f"--> Drawer open ({drawer_pos.max().item():.3f}m). Phase 3 Done!"
                    )
                    ep["phase_three_done"] = True
                    ep["trajectory"] = None

            # Build delta actions from trajectory or hold (zero delta)
            if ep["trajectory"] is not None:
                arm_action = target_state.position[arm_ids_curobo].unsqueeze(0) - robot_entity.data.joint_pos[:, arm_joint_ids_isaac]
                base_action = target_state.position[base_ids_curobo].unsqueeze(0) - robot_entity.data.joint_pos[:, base_joint_ids_isaac]
            else:
                arm_action = torch.zeros(1, len(arm_joint_ids_isaac), device=env.device)
                base_action = torch.zeros(1, len(base_joint_ids_isaac), device=env.device)
        else:
            arm_action = torch.zeros(1, len(arm_joint_ids_isaac), device=env.device)
            base_action = torch.zeros(1, len(base_joint_ids_isaac), device=env.device)

        # ==========================================
        # GRIPPER LOGIC
        # ==========================================
        should_close = (
            ep["phase_two_done"] or ep["phase_three_done"] or (ep["gripper_timer"] > 0)
        )
        gripper_target = GRIPPER_CLOSE_POS if should_close else GRIPPER_OPEN_POS
        gripper_current = robot_entity.data.joint_pos[:, gripper_joint_ids_isaac]
        gripper_action = torch.tensor([[gripper_target, gripper_target]], device=env.device) - gripper_current

        env_actions = torch.cat([arm_action, base_action, gripper_action], dim=-1)

        # ==========================================
        # STEP & RECORD
        # ==========================================
        next_obs, rew, terminated, truncated, _ = env.step(env_actions)

        # Record to the appropriate collector based on current phase
        if not ep["phase_two_done"]:
            collector_reach.add("obs", obs)
            collector_reach.add("actions", env_actions)
            collector_reach.add("rewards", rew)
            collector_reach.add("dones", terminated | truncated)
            collector_reach.add("next_obs", next_obs)
        else:
            collector_pull.add("obs", obs)
            collector_pull.add("actions", env_actions)
            collector_pull.add("rewards", rew)
            collector_pull.add("dones", terminated | truncated)
            collector_pull.add("next_obs", next_obs)

        obs = next_obs
        ep["step_count"] += 1

        # ==========================================
        # PHASE 3 GLOBAL TIMEOUT
        # ==========================================
        if ep["phase_two_done"] and not ep["phase_three_done"]:
            phase3_steps = ep["step_count"] - (
                ep["phase3_start_step"] or ep["step_count"]
            )
            if phase3_steps > 300:
                drawer_pos = cabinet_entity.data.joint_pos[0]
                print(
                    f"--> Phase 3 global timeout at step {ep['step_count']} "
                    f"({phase3_steps} steps into Phase 3). "
                    f"Drawer at {drawer_pos.max().item():.3f}m."
                )
                ep["phase_three_done"] = True
                ep["trajectory"] = None

        # ==========================================
        # SUCCESS CHECK
        # ==========================================
        if ep["phase_three_done"]:
            ep["success_hold_timer"] += 1
            if ep["success_hold_timer"] > SUCCESS_HOLD_STEPS:
                final_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
                displacement = torch.norm(final_pos - initial_handle_pos).item()

                if displacement > HANDLE_DISPLACEMENT_THRESHOLD:
                    print(
                        f"[Record] SUCCESS! Drawer moved {displacement:.3f}m. Saving demo."
                    )
                    collector_reach.flush()
                    collector_pull.flush()
                    demos_collected += 1
                else:
                    print(
                        f"[Record] FAIL. Drawer only moved {displacement:.3f}m. Discarding."
                    )
                    collector_reach.reset_buffer()
                    collector_pull.reset_buffer()
                    demos_failed += 1

                elapsed = time.time() - sim_start_time
                print(
                    f"  [{demos_collected}/{args_cli.num_demos} demos | {demos_failed} failed | {elapsed:.0f}s elapsed]"
                )

                # Reset for next episode
                obs, _ = env.reset()
                randomize_cabinet(
                    cabinet_view,
                    cabinet_entity,
                    default_cabinet_pos,
                    default_cabinet_rot,
                    env.device,
                    args_cli.noise_range,
                )
                phase_targets = compute_phase_targets(
                    target_frame_view, cabinet_view, robot_entity, env.device
                )
                initial_handle_pos = cabinet_entity.data.body_pos_w[
                    0, handle_body_idx
                ].clone()
                ep = reset_episode_state()
                target_pose = phase_targets[ep["current_phase_name"]]
                motion_gen.reset()

    # ==========================================
    # CLEANUP
    # ==========================================
    total_time = time.time() - sim_start_time
    print(
        f"\n[SUMMARY] {demos_collected} demos saved, {demos_failed} failed | {total_time:.0f}s total"
    )
    env.close()
    collector_reach.close()
    collector_pull.close()


if __name__ == "__main__":
    main()
