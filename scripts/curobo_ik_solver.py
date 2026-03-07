import argparse
from pathlib import Path
import os
import numpy as np
import torch
from tqdm import tqdm
from enum import Enum
from typing import Union, Optional
from copy import deepcopy

# from build_env import simulation_app, WorldEngine
from build_env import simulation_app, WorldEngine

# --- Launch Isaac Sim ---
from isaacsim.simulation_app import SimulationApp

# import carb

# simulation_app = SimulationApp(
#     {
#         "headless": False,
#         "width": "1280",
#         "height": "720",
# #     }
# )

from pxr import UsdGeom, Gf, UsdLux, Gf, Usd
import carb


from isaacsim.core.api import World, SimulationContext
from isaacsim.core.api.objects import VisualCuboid, cuboid, sphere
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim, RigidPrim, GeometryPrim
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.torch.rotations import (
    quat_mul,
    quat_from_euler_xyz,
    get_euler_xyz,
)

import omni.kit.commands
import omni.usd


# ------CuRobo Modules------
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.geom.transform import pose_inverse, pose_multiply
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger

# from curobo.util.usd_helper import UsdHelper

from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# from curobo.util_file import (

#     get_world_configs_path,

# )
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


from curobo.util.logger import log_warn


# custom modules
from helper import add_extensions, add_robot_to_scene
from utils.file_util import (
    get_path_of_dir,
    get_filename,
    get_robot_configs_path_own_folder,
    get_assets_path_own_folder,
    get_configs_path_own_folder,
    join_path,
    load_yaml,
)


MotionGenParams = {
    "interpolation_dt": 0.03,
    "collision_cache": {"obb": 10, "mesh": 10},
    "collision_activation_distance": 0.025,
    "maximum_trajectory_dt": 0.25,
    "optimize_dt": True,
    "max_attempts": 4,
    "trajopt_dt": None,
    "trajopt_tsteps": 32,
    "trim_steps": None,
    "enable_finetune_trajopt": False,
}


class GripperStatus(Enum):
    OPEN = 1
    CLOSED = 2


class CuRoBoIKSolver:
    """

    A wrapper class for CuRoBO IK Solver integrated with Isaac Sim ArticulationController.

    """

    def __init__(self, env: WorldEngine) -> None:
        self.env = env

        self.tensor_args = self.env.tensor_args
        self.usd_help = self.env.usd_helper

        self.motion_gen_params = {
            "interpolation_dt": 0.03,
            "collision_cache": {"obb": 10, "mesh": 10},
            "collision_activation_distance": 0.025,
            "maximum_trajectory_dt": 0.25,
            "optimize_dt": False,
            "max_attempts": 4,
            "trajopt_dt": None,
            "trajopt_tsteps": 32,
            "trim_steps": None,
            "enable_finetune_trajopt": True,
        }

        # self.cmd_plan = [None] * self.env.num_env
        self.cmd_idx = [0] * self.env.num_env
        _, self.robot_js_names, _ = self.get_robot_js(0)

        self.robot_camera_joints = self.env.robot_joint_groups.get("camera_joints", [])
        self.robot_base_joints = self.env.robot_joint_groups.get("base_joints", [])
        self.robot_gripper_joints = self.env.robot_joint_groups.get(
            "gripper_joints", []
        )
        self.robot_arm_joints = self.env.robot_joint_groups.get("arm_joints", [])
        self.robot_wrist_joints = self.env.robot_joint_groups.get("wrist_joints", [])

        self.robot_nav_joints = self.robot_base_joints + ["joint_lift"]
        self.robot_manip_joints = self.robot_arm_joints + self.robot_wrist_joints
        self.virtual_base_translation_joints = ["joint_x", "joint_y"]

    def _set_visulized_goal(self) -> None:
        """
        This is only for visualization purpose. It will set several red cubes to test whether the IK
        solver can reach the goal positions without collisions.

        """

        self.vis_target_list = []

        for env_idx in range(self.env.num_env):
            target = cuboid.VisualCuboid(
                "/World/Env_" + str(env_idx) + "/target",
                position=np.array([0.5, 0, 0.5])
                + self.env.env_base_positions[env_idx].cpu().numpy(),
                orientation=np.array([0, 1, 0, 0]),
                color=np.array([1.0, 0, 0]),
                size=0.05,
            )
            self.vis_target_list.append(target)

    def _modify_robot_cfg_for_mode(
        self,
        mode: str = "nav_only",
        robot_cfg: Union[dict, None] = None,
        debug_verbose: bool = False,
        base_movement_soft_constraint: bool = True,
    ) -> dict:

        robot_cfg = deepcopy(robot_cfg)
        if mode == "nav_only":
            robot_cfg["kinematics"]["ee_link"] = "base_link"
            allowed_joints = self.robot_nav_joints
        elif mode == "manip_only":
            robot_cfg["kinematics"]["ee_link"] = "link_grasp_center"
            allowed_joints = self.robot_manip_joints
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported modes are 'nav_only' and 'manip_only'."
            )

        previous_lock_joints = robot_cfg["kinematics"].get("lock_joints", {})

        # Always lock the camera and gripper finger for easier motion planning
        previous_gripper_locked_config = {
            k: previous_lock_joints[k]
            for k in self.robot_gripper_joints
            if k in previous_lock_joints
        }
        previous_camera_locked_config = {
            k: previous_lock_joints[k]
            for k in self.robot_camera_joints
            if k in previous_lock_joints
        }

        # if previous_lock_joints is None: previous_lock_joints = {}
        # --- Lock Arm Joints for Navigation Only ---
        kinematics = robot_cfg.get("kinematics", {})
        cspace = kinematics.get("cspace", {})
        joint_names = cspace.get("joint_names", [])
        retract_config = cspace.get("retract_config", [])  # Default configuration

        lock_joints = {}
        new_cspace = {
            "joint_names": allowed_joints,
            "retract_config": [0.0] * len(allowed_joints),  # Only for allowed joints
            "null_space_weight": [0.1] * len(allowed_joints),  # Only for allowed joints
            "cspace_distance_weight": [1.0]
            * len(allowed_joints),  # Only for allowed joints
            "max_jerk": cspace.get("max_jerk", 500.0),
            "max_acceleration": cspace.get("max_acceleration", 15.0),
        }

        for i, j_name in enumerate(joint_names):
            if j_name not in allowed_joints:
                # Get the value to lock to (from retract/home config)
                val = retract_config[i]
                if hasattr(val, "item"):
                    val = val.item()  # Handle tensor
                lock_joints[j_name] = val

        # Update robot_cfg with locked joints
        # This tells curobo to remove these joints from the optimization problem
        kinematics["lock_joints"] = {
            **lock_joints,
            **previous_gripper_locked_config,
            **previous_camera_locked_config,
        }  # Merge with any previously locked joints for gripper and camera
        kinematics["cspace"] = new_cspace
        robot_cfg["kinematics"] = kinematics

        if debug_verbose:
            print(
                "---------------------DEBUG OUTPUT FOR MODIFIED ROBOT CFG------------------------------------------------"
            )
            print(f"[DEBUG] Modifying robot_cfg for mode: {mode}")
            print(f"[DEBUG] {mode.capitalize()}: Active joints: {allowed_joints}")
            print(
                f"[DEBUG] {mode.capitalize()}: Locked {len(lock_joints)} joints: {list(lock_joints.keys())}"
            )
            print(
                f"[DEBUG] {mode.capitalize()}: \n Updated robot_cfg lock_joints: {robot_cfg['kinematics']['lock_joints']},\n ee_link: {robot_cfg['kinematics']['ee_link']}, \ncspace: {robot_cfg['kinematics']['cspace']}"
            )
            print(
                "---------------------END OF DEBUG OUTPUT FOR MODIFIED ROBOT CFG----------------------------------------"
            )

        return robot_cfg

    def generate_motion_gen(
        self,
        # joints_to_lock: Optional[list]=None,
        world_cfg_list: Union[list | None] = None,
        robot_cfg: Union[dict | None] = None,
        motion_gen_params: Union[dict | None] = None,
        nav_only: bool = False,
        collision_checker_type=CollisionCheckerType.MESH,
        save_log: bool = False,
    ) -> MotionGen:
        """
        Generate CuRoBO MotionGen instance based on the robot and world configurations.

        Args:
            motion_gen_params (dict): Parameters for MotionGen configuration.


        Returns:
            MotionGen: An instance of CuRoBO MotionGen.
        """

        if motion_gen_params is None:
            motion_gen_params = self.motion_gen_params

        if world_cfg_list is None:
            world_cfg_list = self.env.world_config_list

        if robot_cfg is None:
            robot_cfg = self.env.curobo_robot_config

        if nav_only:
            modified_robot_cfg = self._modify_robot_cfg_for_mode(
                mode="nav_only", robot_cfg=robot_cfg, debug_verbose=True
            )
        else:
            modified_robot_cfg = self._modify_robot_cfg_for_mode(
                mode="manip_only", robot_cfg=robot_cfg, debug_verbose=True
            )

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            modified_robot_cfg,
            world_cfg_list,
            self.tensor_args,
            collision_checker_type=collision_checker_type,
            use_cuda_graph=True,
            interpolation_dt=motion_gen_params.get("interpolation_dt", 0.03),
            collision_cache=motion_gen_params.get(
                "collision_cache", {"obb": 10, "mesh": 10}
            ),
            collision_activation_distance=motion_gen_params.get(
                "collision_activation_distance", 0.025
            ),
            maximum_trajectory_dt=motion_gen_params.get("maximum_trajectory_dt", 0.25),
            optimize_dt=motion_gen_params.get("optimize_dt", True),
            trajopt_dt=motion_gen_params.get("trajopt_dt", None),
            trajopt_tsteps=motion_gen_params.get("trajopt_tsteps", 32),
            trim_steps=motion_gen_params.get("trim_steps", None),
            position_threshold=0.001,
            store_ik_debug=save_log,
            store_trajopt_debug=save_log,
        )

        motion_gen = MotionGen(motion_gen_config)
        self.batched_motion_gen = motion_gen
        return motion_gen

    def get_robot_js(self, env_id: int):
        """
        Get the current joint state of a specific robot in the environment.

        Args:
            env_id (int): The environment ID of the robot.

        Returns:
            JointState: The current joint state of the specified robot.
        """

        sim_js_names = self.env.robot_prims[env_id].dof_names
        sim_js = self.env.robot_prims[env_id].get_joints_state()

        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions).view(1, -1),
            velocity=self.tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities).view(1, -1)
            * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            joint_names=sim_js_names,
        )

        return cu_js, sim_js_names, sim_js

    def get_gripper_status(self, env_id: int) -> GripperStatus:
        """
        Get the current gripper status of a specific robot in the environment.

        Args:
            env_id (int): The environment ID of the robot.

        Returns:
            GripperStatus: The current gripper status of the specified robot.
        """

        sim_js = self.env.robot_prims[env_id].get_joints_state()

        gripper_status = {}

        for joint_names in self.robot_gripper_joints:
            joint_index = self.env.robot_prims[env_id].get_dof_index(joint_names)
            joint_position = sim_js.positions[joint_index]
            if joint_position > 0.01:
                gripper_status[joint_names] = {
                    "status": GripperStatus.OPEN,
                    "position": joint_position,
                }

            else:
                gripper_status[joint_names] = {
                    "status": GripperStatus.CLOSED,
                    "position": joint_position,
                }

        return gripper_status

    def get_current_robot_full_js(self) -> JointState:
        """
        Get the current full joint state of all robots in the environment.

        Returns:
            JointState: The current full joint state of all robots.
        """

        full_js, sim_js_name, _ = self.get_robot_js(0)

        for i in range(1, self.env.num_env):
            cu_js, cu_name, _ = self.get_robot_js(i)
            full_js = full_js.stack(cu_js)

        return full_js

    def ik_solver_batch(
        self,
        ik_goal: Pose,
        motiongen: Union[MotionGen, None] = None,
        nav_only: bool = False,
    ):
        """
        Given a batch of ik_goal for multiple envs, solve IK for all envs in batch.
        Args:
            ik_goal (Pose): The desired end-effector poses for the robots.The postion shape should be [num_env, 3] and quaternion shape should be [num_env, 4] and both should be tensors.

        """
        cmd_plan = [None] * self.env.num_env
        if not isinstance(ik_goal.position, torch.Tensor):
            ik_goal.position = self.tensor_args.to_device(ik_goal.position)
            carb.log_warn(
                f"Pose.position should be torch.Tensor, currrently the type is {type(ik_goal.position)}.Converted ik_goal.position to tensor."
            )
        if not isinstance(ik_goal.quaternion, torch.Tensor):
            ik_goal.quaternion = self.tensor_args.to_device(ik_goal.quaternion)
            carb.log_warn(
                f"Pose.quaternion should be torch.Tensor, currrently the type is {type(ik_goal.quaternion)}.Converted ik_goal.quaternion to tensor."
            )

        if motiongen is None:
            motiongen = self.batched_motion_gen

        plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            need_graph_success=True,
            max_attempts=self.motion_gen_params.get("max_attempts", 60),
            enable_finetune_trajopt=self.motion_gen_params.get(
                "enable_finetune_trajopt", True
            ),
        )

        _, sim_js_names, _ = self.get_robot_js(0)
        full_js = self.get_current_robot_full_js()

        full_js = full_js.get_ordered_joint_state(motiongen.kinematics.joint_names)
        result = motiongen.plan_batch_env(full_js, ik_goal, plan_config.clone())

        # print(f"[INFO] IK Solver Batch Planning Result: {result.status}, Successes: {result.success}")

        if torch.count_nonzero(result.success) > 0:
            trajs = (
                result.get_paths()
            )  # Get the joint space trajectories for each robot

            for s in range(len(result.success)):
                if result.success[s]:
                    cmd_plan[s] = motiongen.get_full_js(trajs[s])
                    # get only joint names that are in both:
                    idx_list = []
                    common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan[s].joint_names:
                            idx_list.append(self.env.robot_prims[s].get_dof_index(x))
                            common_js_names.append(x)

                    cmd_plan[s] = cmd_plan[s].get_ordered_joint_state(common_js_names)

            return {"plan_result": result, "cmd_plan": cmd_plan, "idx_list": idx_list}

        else:
            print("[WARN] IK Solver Batch Planning Failed: No successful plans found.")
            return {"plan_result": result.status, "cmd_plan": None, "idx_list": None}

    def visualize_gripper_pose(
        self, pose: Pose, prefix: str = "pose", size: float = 0.15
    ):
        """
        Draw persistent coordinate frames for a batch of poses using VisualCuboids. Uses torch tensors primarily.

        Args:
            stage: USD Stage
            pose (Pose): The pose to visualize.
            prefix (str): Unique prefix for the USD prims
            size (float): Length of the axis lines
        """
        num_poses = pose.position.shape[0]
        thickness = 0.01

        for i in range(num_poses):
            p = pose.position[i : i + 1]  # Keep as tensor, slice to maintain dim
            q = pose.quaternion[i : i + 1]  # Keep as tensor

            base_path = f"/World/Env_{i}/{prefix}"

            # Create root XForm for the frame
            UsdGeom.Xform.Define(self.env.stage, base_path)
            XFormPrim(prim_paths_expr=base_path, translations=p, orientations=q)

            # X-Axis (Red)
            VisualCuboid(
                prim_path=f"{base_path}/x_axis",
                translation=self.env.tensor_args.to_device(np.array([size / 2, 0, 0])),
                scale=self.env.tensor_args.to_device(
                    np.array([size, thickness, thickness])
                ),
                color=self.env.tensor_args.to_device(np.array([1.0, 0, 0])),
            )

            # Y-Axis (Green)
            VisualCuboid(
                prim_path=f"{base_path}/y_axis",
                translation=self.env.tensor_args.to_device(np.array([0, size / 2, 0])),
                scale=self.env.tensor_args.to_device(
                    np.array([thickness, size, thickness])
                ),
                color=self.env.tensor_args.to_device(np.array([0, 1.0, 0])),
            )

            # Z-Axis (Blue)
            VisualCuboid(
                prim_path=f"{base_path}/z_axis",
                translation=self.env.tensor_args.to_device(np.array([0, 0, size / 2])),
                scale=self.env.tensor_args.to_device(
                    np.array([thickness, thickness, size])
                ),
                color=self.env.tensor_args.to_device(np.array([0, 0, 1.0])),
            )

    def get_relative_pose(self, target_pose: Pose, reference_pose: Pose) -> Pose:
        """
        Compute the relative pose P_rel of the target with respect to a reference frame.
        P_target=P_reference * P_rel => P_rel = P_reference^-1 * P_target

        Args:
            target_pose (Pose): The target pose in world coordinates.
            reference_pose (Pose): The reference pose in world coordinates.

        Returns:
            Pose: The relative pose of the target with respect to the reference frame.
        """
        # 1. Compute inverse of P1 (P1^-1)
        # pose_inverse takes (position, quaternion) tensors
        p1_inv_pos, p1_inv_quat = pose_inverse(
            reference_pose.position, reference_pose.quaternion
        )

        # 2. Multiply P1^-1 * P2
        # pose_multiply takes (pos1, quat1, pos2, quat2)
        rel_pos, rel_quat = pose_multiply(
            p1_inv_pos, p1_inv_quat, target_pose.position, target_pose.quaternion
        )

        return Pose(
            position=self.tensor_args.to_device(rel_pos),
            quaternion=self.tensor_args.to_device(rel_quat),
        )

    def _robot_initialize(self, max_effort=5000):
        for robot in self.env.robot_prims:
            robot._articulation_view.initialize()

            j_names = self.env.curobo_robot_config["kinematics"]["cspace"][
                "joint_names"
            ]
            default_config = self.env.curobo_robot_config["kinematics"]["cspace"][
                "retract_config"
            ]
            idx_list = [robot.get_dof_index(x) for x in j_names]

            if not isinstance(default_config, torch.Tensor):
                default_config = self.tensor_args.to_device(default_config)
            if not isinstance(idx_list, torch.Tensor):
                idx_list = self.tensor_args.to_device(idx_list)
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=self.tensor_args.to_device(
                    [max_effort for i in range(len(idx_list))]
                ),
                joint_indices=idx_list,
            )

            # --- Added Stiffness and Damping ---
            # Create lists for Kp and Kd
            kps = []
            kds = []

            for name in j_names:
                if "joint_arm_l" in name:  # Arm extension joints
                    kps.append(10000.0)
                    kds.append(100.0)
                elif "joint_lift" in name:  # Lift joint
                    kps.append(10000.0)
                    kds.append(100.0)
                elif "wrist" in name:  # Wrist joints
                    kps.append(1000.0)
                    kds.append(50.0)
                elif name in [
                    "joint_x",
                    "joint_y",
                    "joint_rot_z",
                ]:  # Virtual base joints
                    kps.append(10000.0)  # High stiffness for precise base positioning
                    kds.append(200.0)  # High damping to prevent oscillation
                else:  # Default for other joints (head, gripper, etc.)
                    kps.append(800.0)
                    kds.append(40.0)

            # Apply gains
            robot._articulation_view.set_gains(
                kps=self.tensor_args.to_device(kps),
                kds=self.tensor_args.to_device(kds),
                joint_indices=idx_list,
            )
            # -----------------------------------

        return idx_list


def test_ik_solver_modulized(
    env: WorldEngine,
    ik_solver: CuRoBoIKSolver,
):

    ### Get Target
    target_list = []

    for env_idx in range(env.num_env):
        target = cuboid.VisualCuboid(
            "/World/Env_" + str(env_idx) + "/target",
            position=np.array([0.5, 0, 0.5])
            + env.env_base_positions[env_idx].cpu().numpy(),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

    setup_curobo_logger("warn")
    ####

    motion_gen = ik_solver.generate_motion_gen()

    robot_cfg = env.curobo_robot_config
    # # world_cfg_list=env.world_config_list
    # j_names=robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config=robot_cfg["kinematics"]["cspace"]["retract_config"]
    print("Curobo is Ready")

    add_extensions(simulation_app, False)

    # robot_world_config= RobotWorldConfig.load_from_config(
    #     env.curobo_robot_config, world_cfg_list, collision_activation_distance=0.2
    # )

    # model = RobotWorld(robot_world_config)

    cmd_plan = [None] * env.num_env
    prev_goal = None
    past_goal = None
    i = 0
    art_controllers = [r.get_articulation_controller() for r in env.robot_prims]

    cmd_idx = 0
    while simulation_app.is_running():
        env.world.step(render=True)

        if not env.world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = env.world.current_time_step_index

        if step_index <= 10:
            idx_list = ik_solver._robot_initialize(max_effort=5000)

        if step_index < 20:
            cu_js, sim_js_names, js = ik_solver.get_robot_js(0)
            # print(f"[INFO] env_id: 0, \n cu_js  {cu_js}, \n sim_js_names: {sim_js_names}, \n js: {js}")
            continue

        # Get current target poses
        sp_buffer = []
        sq_buffer = []

        for target in target_list:
            sph_position, sph_orientation = target.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        # Convert lists to tensors
        sp_buffer = torch.stack(sp_buffer)  # Shape: [num_env, 3]
        sq_buffer = torch.stack(sq_buffer)  # Shape: [num_env, 4]

        ik_goal = Pose(
            position=env.tensor_args.to_device(sp_buffer),
            quaternion=env.tensor_args.to_device(sq_buffer),
        )  # goal specification
        # print(f"[DEBUG] ik_goal.position: {ik_goal.position}, ik_goal.quaternion: {ik_goal.quaternion}")

        # Get Pose tests:
        # 1. env.get_poses_from_prim, using USD, should be deprecated soon
        print(
            f"[DEBUG] Robot Current pose with get_poses_from_prim: {env.get_poses_from_prim('/World/Env_1/Robot_2/link_lift')}"
        )

        # 2. RigidPrim.get_world_poses, using IsaacSim API, used as reference
        lift_link = RigidPrim("/World/Env_1/Robot_2/link_lift")
        print(
            f"[DEBUG] Robot Current pose with RigidPrim.get_world_pose {lift_link.get_world_poses()}, type: {type(lift_link.get_world_poses())}"
        )
        # Example output: (tensor([[0.0396, 0.3305, 0.5580]], device='cuda:0'), tensor([[ 2.6526e-01,  1.4086e-06, -2.2972e-06,  9.6418e-01]], device='cuda:0'))

        # 3. Self-implemented get_prim_local_pose, using IsaacSim API internally, should be consistent with 2.
        print(
            f"[DEBUG] Robot Current pose with get_prim_local_pose: {env.get_prim_local_pose('/World/Env_1/Robot_2/link_lift')}"
        )

        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()

        prev_distance = ik_goal.distance(prev_goal)
        past_distance = ik_goal.distance(past_goal)

        # print(f"prev_distance: {prev_distance}, past_distance: {past_distance}")
        result = None

        full_js = ik_solver.get_current_robot_full_js()

        # print(f"=== Planning Condition Check ===")
        # print(f"prev_distance > 1e-2: position={torch.sum(prev_distance[0] > 1e-2)}, orientation={torch.sum(prev_distance[1] > 1e-2)}")
        # print(f"past_distance == 0: position={torch.sum(past_distance[0]) == 0.0}, orientation={torch.sum(past_distance[1] == 0.0)}")
        # print(f"max velocity: {torch.max(torch.abs(full_js.velocity))}")
        # print(f"all cmd_plan None: {all(p is None for p in cmd_plan)}")
        # print(f"cmd_plan states: {[p is not None for p in cmd_plan]}")

        if (
            (
                torch.sum(prev_distance[0] > 1e-2) or torch.sum(prev_distance[1] > 1e-2)
            )  # has the goal changed significantly since the last plan?
            and (
                torch.sum(past_distance[0]) == 0.0
                and torch.sum(past_distance[1] == 0.0)
            )  # was the target stationary in the previous step?
            and torch.max(torch.abs(full_js.velocity))
            < 0.2  # are the robots currently stopped or moving very slowly?
            and all(
                p is None for p in cmd_plan
            )  # are there no active plans for all robots?
        ):  # Overall: should we plan for all robots now?

            plan_result = ik_solver.ik_solver_batch(ik_goal, motiongen=motion_gen)
            result = plan_result["plan_result"]
            cmd_plan = plan_result["cmd_plan"]
            idx_list = plan_result["idx_list"]  # Get the correct indices
            prev_goal.copy_(ik_goal)  # update the previous goal to current goal
            cmd_idx = 0

        # print(f"length of cmd_plan: {len(cmd_plan[0])}, cmd_idx: {cmd_idx}")

        for s in range(len(cmd_plan)):

            if cmd_plan[s] is not None and cmd_idx < len(cmd_plan[s].position):
                cmd_state = cmd_plan[s][cmd_idx]

                # get full dof state
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )

                # print(art_action)
                # set desired joint angles obtained from IK:
                art_controllers[s].apply_action(art_action)
            elif result is not None and not result.success[s]:
                cmd_plan[s] = None
                carb.log_warn(
                    f"For robot{s}, Plan did not converge to a solution: {result.status}"
                )
            else:
                cmd_plan[s] = None
                carb.log_info(f"For robot{s}, No active plan.")

        cmd_idx += 1
        past_goal.copy_(ik_goal)

        for _ in range(2):
            env.world.step(render=False)


def test_ik_solver_modulized_grasping(
    env: WorldEngine,
    ik_solver: CuRoBoIKSolver,
):

    ### Get Target
    target_list = []

    for env_idx in range(env.num_env):
        target = cuboid.VisualCuboid(
            "/World/Env_" + str(env_idx) + "/target",
            position=np.array([0.5, 0, 0.5])
            + env.env_base_positions[env_idx].cpu().numpy(),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

    # Helper function inspired by sim_skills.py
    def align_gripper_to_object(q_obj, q_gripper_down):
        """
        q_obj: N * [w, x, y, z] of the target object
        q_gripper_down: N * [w, x, y, z] default gripper orientation (pointing down)
        """

        # Extract Yaw (w, 0, 0, z)
        q_yaw_only = torch.zeros_like(q_obj)
        q_yaw_only[:, 0] = q_obj[:, 0]  # w
        q_yaw_only[:, 3] = q_obj[:, 3]  # z

        # Normalize
        norms = torch.norm(q_yaw_only, dim=1, keepdim=True)

        # Avoid divide by zero
        # Use torch.where to handle batched inputs safely
        q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_obj.device).expand_as(
            q_yaw_only
        )
        mask = norms < 1e-6

        # Fallback to identity if norm is too small, otherwise normalize
        # Adding epsilon to denominator to prevent NaN generation in the division branch
        q_yaw_only = torch.where(mask, q_identity, q_yaw_only / (norms + 1e-9))

        q_yaw_rev = torch.zeros_like(q_yaw_only)
        q_yaw_rev[:, 0] = -q_yaw_only[:, 3]  # New w = -old z
        q_yaw_rev[:, 3] = q_yaw_only[:, 0]  # New z = old w

        q_final = quat_mul(q_yaw_only, q_gripper_down.expand_as(q_yaw_only))
        q_final_flipped = quat_mul(q_yaw_rev, q_gripper_down.expand_as(q_yaw_only))
        q_obj_gripper_down = quat_mul(q_obj, q_gripper_down.expand_as(q_obj))

        return q_final, q_final_flipped, q_obj_gripper_down

    def align_gripper_to_object_euler(rotation_obj_z, return_in_euler=True):
        # won't use the rotation on x and y axis for top grasp
        rotation_euler_x_obb = torch.zeros_like(rotation_obj_z)
        rotation_euler_y_obb = torch.zeros_like(rotation_obj_z)

        rotation_euler_y_obb[:] = (
            0.5 * np.pi
        )  # Rotate 90 degrees around y-axis to make gripper point downwards

        if return_in_euler:
            return rotation_euler_x_obb, rotation_euler_y_obb, rotation_obj_z

        else:
            target_quat = quat_from_euler_xyz(
                rotation_euler_x_obb, rotation_euler_y_obb, rotation_obj_z
            )
            return target_quat

    def pre_grasp_pose_generation(pose, offset=0.2):

        pre_grasp_pose_position = pose.position + env.tensor_args.to_device(
            torch.tensor([[0.0, 0.0, offset]]).expand_as(pose.position)
        )

        euler_x, euler_y, euler_z = get_euler_xyz(pose.quaternion)

        # q_gripper_down= env.tensor_args.to_device(torch.tensor([0.7071, 0.0, -0.7071, 0.0])) # w,x,y,z
        # q_gripper_down= env.tensor_args.to_device(torch.tensor([1, 0.0, 0.0, 0.0])) # w,x,y,z

        pre_grasp_pose_quaternion_obj_gripper_down = align_gripper_to_object_euler(
            euler_z, return_in_euler=False
        )

        pre_grasp_pose = Pose(
            position=pre_grasp_pose_position,
            quaternion=pre_grasp_pose_quaternion_obj_gripper_down,
        )

        return pre_grasp_pose

    setup_curobo_logger("warn")
    ####

    motion_gen = ik_solver.generate_motion_gen()

    robot_cfg = env.curobo_robot_config
    # # world_cfg_list=env.world_config_list
    # j_names=robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config=robot_cfg["kinematics"]["cspace"]["retract_config"]
    print("Curobo is Ready")

    add_extensions(simulation_app, False)

    # robot_world_config= RobotWorldConfig.load_from_config(
    #     env.curobo_robot_config, world_cfg_list, collision_activation_distance=0.2
    # )

    # model = RobotWorld(robot_world_config)

    cmd_plan = [None] * env.num_env
    prev_goal = None
    past_goal = None
    i = 0
    art_controllers = [r.get_articulation_controller() for r in env.robot_prims]

    cmd_idx = 0
    while simulation_app.is_running():
        env.world.step(render=True)

        if not env.world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = env.world.current_time_step_index

        if step_index <= 10:
            idx_list = ik_solver._robot_initialize(max_effort=5000)

        if step_index < 20:
            cu_js, sim_js_names, js = ik_solver.get_robot_js(0)
            # print(f"[INFO] env_id: 0, \n cu_js  {cu_js}, \n sim_js_names: {sim_js_names}, \n js: {js}")
            continue

        # Get current target poses
        sp_buffer = []
        sq_buffer = []

        for target in target_list:
            sph_position, sph_orientation = target.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        # Convert lists to tensors
        sp_buffer = torch.stack(sp_buffer)  # Shape: [num_env, 3]
        sq_buffer = torch.stack(sq_buffer)  # Shape: [num_env, 4]

        ik_goal = Pose(
            position=env.tensor_args.to_device(sp_buffer),
            quaternion=env.tensor_args.to_device(sq_buffer),
        )  # goal specification

        # ik_goal= pre_grasp_pose_generation(ik_goal, offset=0.2)

        ik_solver.visualize_gripper_pose(ik_goal, prefix="ik_goal", size=0.2)

        current_base_pose_env_0 = env.get_prim_local_pose(
            f"/World/Env_0/Robot_0/base_link"
        )
        current_base_pose_env_1 = env.get_prim_local_pose(
            f"/World/Env_1/Robot_1/base_link"
        )
        print(f"[DEBUG] current_base_pose_env_0: {current_base_pose_env_0}")
        print(f"[DEBUG] current_base_pose_env_1: {current_base_pose_env_1}")

        # print(f"[DEBUG] ik_goal.position: {ik_goal.position}, ik_goal.quaternion: {ik_goal.quaternion}")

        # # Get Pose tests:
        # # 1. env.get_poses_from_prim, using USD, should be deprecated soon
        # print(f"[DEBUG] Robot Current pose with get_poses_from_prim: {env.get_poses_from_prim('/World/Env_1/Robot_2/link_lift')}")

        # # 2. RigidPrim.get_world_poses, using IsaacSim API, used as reference
        # lift_link=RigidPrim("/World/Env_1/Robot_2/link_lift")
        # print(f"[DEBUG] Robot Current pose with RigidPrim.get_world_pose {lift_link.get_world_poses()}, type: {type(lift_link.get_world_poses())}")
        # # Example output: (tensor([[0.0396, 0.3305, 0.5580]], device='cuda:0'), tensor([[ 2.6526e-01,  1.4086e-06, -2.2972e-06,  9.6418e-01]], device='cuda:0'))

        # # 3. Self-implemented get_prim_local_pose, using IsaacSim API internally, should be consistent with 2.
        # print(f"[DEBUG] Robot Current pose with get_prim_local_pose: {env.get_prim_local_pose('/World/Env_1/Robot_2/link_lift')}")

        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()

        prev_distance = ik_goal.distance(prev_goal)
        past_distance = ik_goal.distance(past_goal)

        # print(f"prev_distance: {prev_distance}, past_distance: {past_distance}")
        result = None

        full_js = ik_solver.get_current_robot_full_js()

        # print(f"=== Planning Condition Check ===")
        # print(f"prev_distance > 1e-2: position={torch.sum(prev_distance[0] > 1e-2)}, orientation={torch.sum(prev_distance[1] > 1e-2)}")
        # print(f"past_distance == 0: position={torch.sum(past_distance[0]) == 0.0}, orientation={torch.sum(past_distance[1] == 0.0)}")
        # print(f"max velocity: {torch.max(torch.abs(full_js.velocity))}")
        # print(f"all cmd_plan None: {all(p is None for p in cmd_plan)}")
        # print(f"cmd_plan states: {[p is not None for p in cmd_plan]}")

        if (
            (
                torch.sum(prev_distance[0] > 1e-2) or torch.sum(prev_distance[1] > 1e-2)
            )  # has the goal changed significantly since the last plan?
            and (
                torch.sum(past_distance[0]) == 0.0
                and torch.sum(past_distance[1] == 0.0)
            )  # was the target stationary in the previous step?
            and torch.max(torch.abs(full_js.velocity))
            < 0.2  # are the robots currently stopped or moving very slowly?
            and all(
                p is None for p in cmd_plan
            )  # are there no active plans for all robots?
        ):  # Overall: should we plan for all robots now?

            plan_result = ik_solver.ik_solver_batch(ik_goal, motiongen=motion_gen)
            result = plan_result["plan_result"]
            cmd_plan = plan_result["cmd_plan"]
            idx_list = plan_result["idx_list"]  # Get the correct indices
            prev_goal.copy_(ik_goal)  # update the previous goal to current goal
            cmd_idx = 0

        # print(f"length of cmd_plan: {len(cmd_plan[0])}, cmd_idx: {cmd_idx}")

        if cmd_plan is None:
            carb.log_warn(
                f"No valid cmd_plan generated from IK solver for all robots. The results were: {result}"
            )

        for s in range(len(cmd_plan)):

            if cmd_plan[s] is not None and cmd_idx < len(cmd_plan[s].position):
                cmd_state = cmd_plan[s][cmd_idx]

                # get full dof state
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )

                # print(art_action)
                # set desired joint angles obtained from IK:
                art_controllers[s].apply_action(art_action)
            elif result is not None and not result.success[s]:
                cmd_plan[s] = None
                carb.log_warn(
                    f"For robot{s}, Plan did not converge to a solution: {result.status}"
                )
            else:
                cmd_plan[s] = None
                carb.log_info(f"For robot{s}, No active plan.")

        cmd_idx += 1
        past_goal.copy_(ik_goal)

        for _ in range(2):
            env.world.step(render=False)


def test_ik_solver_modulized_navigation(
    env: WorldEngine,
    ik_solver: CuRoBoIKSolver,
):

    ### Get Target
    target_list = []

    for env_idx in range(env.num_env):
        target = cuboid.VisualCuboid(
            "/World/Env_" + str(env_idx) + "/target",
            position=np.array([0.0, 0, 0.5])
            + env.env_base_positions[env_idx].cpu().numpy(),
            orientation=np.array([1, 0, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)

    setup_curobo_logger("warn")
    ####

    # navigation_robot_config=load_yaml(os.path.join(get_robot_configs_path_own_folder(),"stretch_fake_joint_nav.yml"))
    # navigation_robot_config=navigation_robot_config["robot_cfg"]
    # navigation_robot_config["kinematics"]["external_asset_path"] = get_assets_path_own_folder()
    # navigation_robot_config["kinematics"]["external_robot_configs_path"]= get_robot_configs_path_own_folder()
    # motion_gen= ik_solver.generate_motion_gen(robot_cfg=navigation_robot_config)

    motion_gen = ik_solver.generate_motion_gen(nav_only=True)

    # # world_cfg_list=env.world_config_list
    # j_names=robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config=robot_cfg["kinematics"]["cspace"]["retract_config"]
    print("Curobo is Ready")

    add_extensions(simulation_app, False)

    # robot_world_config= RobotWorldConfig.load_from_config(
    #     env.curobo_robot_config, world_cfg_list, collision_activation_distance=0.2
    # )

    # model = RobotWorld(robot_world_config)

    def base_pose_generation(pose: Pose):
        base_pose_position = pose.position.clone()
        base_pose_position[:, 2] = 0.0  # set z to 0 for base

        base_pose_quaternion = pose.quaternion.clone()

        base_pose = Pose(
            position=base_pose_position,
            quaternion=base_pose_quaternion,
        )
        return base_pose

    cmd_plan = [None] * env.num_env
    prev_goal = None
    past_goal = None
    i = 0
    art_controllers = [r.get_articulation_controller() for r in env.robot_prims]

    cmd_idx = 0
    while simulation_app.is_running():
        env.world.step(render=True)

        if not env.world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = env.world.current_time_step_index

        if step_index <= 10:
            idx_list = ik_solver._robot_initialize(max_effort=5000)

        if step_index < 20:
            cu_js, sim_js_names, js = ik_solver.get_robot_js(0)
            # print(f"[INFO] env_id: 0, \n cu_js  {cu_js}, \n sim_js_names: {sim_js_names}, \n js: {js}")
            continue

        # Get current target poses
        sp_buffer = []
        sq_buffer = []

        for target in target_list:
            sph_position, sph_orientation = target.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        # Convert lists to tensors
        sp_buffer = torch.stack(sp_buffer)  # Shape: [num_env, 3]
        sq_buffer = torch.stack(sq_buffer)  # Shape: [num_env, 4]

        ik_goal = Pose(
            position=env.tensor_args.to_device(sp_buffer),
            quaternion=env.tensor_args.to_device(sq_buffer),
        )  # goal specification

        ik_goal = base_pose_generation(ik_goal)

        # print(f"[DEBUG] ik_goal for navigation: position={ik_goal.position}, quaternion={ik_goal.quaternion}\n [DEBUG] Robot Base pose in Env 0: {env.get_prim_local_pose(f'/World/Env_0/Robot_0/base_link')} \n [DEBUG] Robot Base pose in Env 1: {env.get_prim_local_pose(f'/World/Env_1/Robot_1/base_link')}")
        ik_solver.visualize_gripper_pose(ik_goal, prefix="ik_goal", size=0.2)

        # print(f"[DEBUG] ik_goal.position: {ik_goal.position}, ik_goal.quaternion: {ik_goal.quaternion}")

        # # Get Pose tests:
        # # 1. env.get_poses_from_prim, using USD, should be deprecated soon
        # print(f"[DEBUG] Robot Current pose with get_poses_from_prim: {env.get_poses_from_prim('/World/Env_1/Robot_2/link_lift')}")

        # # 2. RigidPrim.get_world_poses, using IsaacSim API, used as reference
        # lift_link=RigidPrim("/World/Env_1/Robot_2/link_lift")
        # print(f"[DEBUG] Robot Current pose with RigidPrim.get_world_pose {lift_link.get_world_poses()}, type: {type(lift_link.get_world_poses())}")
        # # Example output: (tensor([[0.0396, 0.3305, 0.5580]], device='cuda:0'), tensor([[ 2.6526e-01,  1.4086e-06, -2.2972e-06,  9.6418e-01]], device='cuda:0'))

        # # 3. Self-implemented get_prim_local_pose, using IsaacSim API internally, should be consistent with 2.
        # print(f"[DEBUG] Robot Current pose with get_prim_local_pose: {env.get_prim_local_pose('/World/Env_1/Robot_2/link_lift')}")

        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()

        prev_distance = ik_goal.distance(prev_goal)
        past_distance = ik_goal.distance(past_goal)

        # print(f"prev_distance: {prev_distance}, past_distance: {past_distance}")
        result = None

        full_js = ik_solver.get_current_robot_full_js()

        # print(f"=== Planning Condition Check ===")
        # print(f"prev_distance > 1e-2: position={torch.sum(prev_distance[0] > 1e-2)}, orientation={torch.sum(prev_distance[1] > 1e-2)}")
        # print(f"past_distance == 0: position={torch.sum(past_distance[0]) == 0.0}, orientation={torch.sum(past_distance[1] == 0.0)}")
        # print(f"max velocity: {torch.max(torch.abs(full_js.velocity))}")
        # print(f"all cmd_plan None: {all(p is None for p in cmd_plan)}")
        # print(f"cmd_plan states: {[p is not None for p in cmd_plan]}")

        if (
            (
                torch.sum(prev_distance[0] > 1e-2) or torch.sum(prev_distance[1] > 1e-2)
            )  # has the goal changed significantly since the last plan?
            and (
                torch.sum(past_distance[0]) == 0.0
                and torch.sum(past_distance[1] == 0.0)
            )  # was the target stationary in the previous step?
            and torch.max(torch.abs(full_js.velocity))
            < 0.2  # are the robots currently stopped or moving very slowly?
            and all(
                p is None for p in cmd_plan
            )  # are there no active plans for all robots?
        ):  # Overall: should we plan for all robots now?

            plan_result = ik_solver.ik_solver_batch(ik_goal, motiongen=motion_gen)
            result = plan_result["plan_result"]
            cmd_plan = plan_result["cmd_plan"]
            idx_list = plan_result["idx_list"]  # Get the correct indices
            prev_goal.copy_(ik_goal)  # update the previous goal to current goal
            cmd_idx = 0

        # print(f"length of cmd_plan: {len(cmd_plan[0])}, cmd_idx: {cmd_idx}")

        if cmd_plan is None:
            carb.log_warn(
                f"No valid cmd_plan generated from IK solver for all robots. The results were: {result}"
            )

        for s in range(len(cmd_plan)):

            if cmd_plan[s] is not None and cmd_idx < len(cmd_plan[s].position):
                cmd_state = cmd_plan[s][cmd_idx]

                # get full dof state
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )

                print(art_action)

                # set desired joint angles obtained from IK:
                art_controllers[s].apply_action(art_action)
            elif result is not None and not result.success[s]:
                cmd_plan[s] = None
                carb.log_warn(
                    f"For robot{s}, Plan did not converge to a solution: {result.status}"
                )
            else:
                cmd_plan[s] = None
                carb.log_info(f"For robot{s}, No active plan.")

        cmd_idx += 1
        past_goal.copy_(ik_goal)

        for _ in range(2):
            env.world.step(render=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_module", type=str, default="grasping")
    args = parser.parse_args()
    world_engine = WorldEngine(
        simulation_app=simulation_app,
        config_path=join_path(
            get_configs_path_own_folder(), "./world/3_table_stretch.yml"
        ),
        device="cuda:0",
        robot_base_name="Stretch",
        num_env=2,
        vertical_offset=0.0,
        grid_spacing=8.0,
    )

    ik_solver = CuRoBoIKSolver(world_engine)
    # test_ik_solver_batched(world_engine)

    # test_ik_solver_modulized(world_engine,ik_solver)

    if args.test_module == "grasping":
        test_ik_solver_modulized_grasping(world_engine, ik_solver)
    elif args.test_module == "nav":
        test_ik_solver_modulized_navigation(world_engine, ik_solver)
    else:
        print(
            """unknown test_module, choose between \"grasping\" or \"nav\", exiting......... """
        )

    print("Shutting down...")
    simulation_app.close()
