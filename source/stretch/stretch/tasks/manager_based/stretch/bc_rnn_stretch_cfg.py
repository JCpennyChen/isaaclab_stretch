from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import mdp as isaac_mdp
from config.stretch_cfg import STRETCH_CFG

# ==========================================
# Shared joint name constants
# ==========================================
ARM_JOINT_NAMES = [
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]
BASE_JOINT_NAMES = ["joint_x", "joint_y", "joint_rot_z"]
GRIPPER_JOINT_NAMES = ["joint_gripper_finger_left", "joint_gripper_finger_right"]
HEAD_JOINT_NAMES = ["joint_head_pan", "joint_head_tilt"]

# Head joint limits from URDF
HEAD_PAN_LIMITS = (-3.9, 1.5)
HEAD_TILT_LIMITS = (-1.53, 0.79)


def compute_head_look_at(robot_entity, cabinet_entity, head_body_idx, handle_body_idx, head_joint_ids):
    """Compute head pan/tilt to look at the drawer handle.

    Uses the robot base frame to decompose the direction into pan/tilt angles.

    Args:
        robot_entity: The robot articulation from the scene.
        cabinet_entity: The cabinet articulation from the scene.
        head_body_idx: Index of link_head in robot body list.
        handle_body_idx: Index of drawer_handle_top in cabinet body list.
        head_joint_ids: Joint indices for [head_pan, head_tilt].

    Returns:
        head_target: (2,) tensor of [pan, tilt] absolute joint positions.
        head_delta: (1, 2) tensor of delta from current head position.
    """
    from isaaclab.utils.math import quat_apply, quat_inv

    head_pos_w = robot_entity.data.body_pos_w[0, head_body_idx]
    handle_pos_w = cabinet_entity.data.body_pos_w[0, handle_body_idx]
    robot_quat_w = robot_entity.data.root_state_w[0, 3:7]

    dir_w = handle_pos_w - head_pos_w
    dir_base = quat_apply(
        quat_inv(robot_quat_w.unsqueeze(0)), dir_w.unsqueeze(0)
    )[0]

    pan = torch.clamp(torch.atan2(dir_base[1], dir_base[0]), *HEAD_PAN_LIMITS)
    horiz = torch.sqrt(dir_base[0] ** 2 + dir_base[1] ** 2)
    tilt = torch.clamp(torch.atan2(dir_base[2], horiz), *HEAD_TILT_LIMITS)

    head_target = torch.stack([pan, tilt])
    current_head = robot_entity.data.joint_pos[0, head_joint_ids]
    head_delta = (head_target - current_head).unsqueeze(0)
    return head_target, head_delta


def reward_cabinet_opening_proportional(
    env, cabinet_cfg: SceneEntityCfg, max_open: float = 0.4
):
    door_pos = env.scene[cabinet_cfg.name].data.joint_pos[:, cabinet_cfg.joint_ids]
    normalized_pos = torch.clamp(door_pos / max_open, min=0.0, max=1.0)
    return torch.sum(torch.pow(normalized_pos, 2), dim=-1)


def handle_rel_pos(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    asset_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids[0]]
    target_pos = env.scene[target_cfg.name].data.body_pos_w[:, target_cfg.body_ids[0]]
    return target_pos - asset_pos


@configclass
class StretchSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with the Stretch robot."""

    # Ground plane
    bg_env = AssetBaseCfg(
        prim_path="/World/Env",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/johnchen/SharedSSD/JohnChen/stretch/assets/ground_plane/default_environment.usd"
        ),
    )

    # Cabinet
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/johnchen/SharedSSD/JohnChen/stretch/assets/object/sektion_cabinet.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -1.5, 0.39),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["door_.*", "drawer_.*"],
                effort_limit=50.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    # Stretch Robot
    robot = STRETCH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # D435 camera mounted on the Stretch's head (child of camera_link)
    # Matches the real robot's head camera position and orientation.
    # The camera_link prim comes from the URDF; we spawn the sensor as a child.
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link/HeadCamera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),  # align with D435 optical axis
            convention="ros",
        ),
        width=640,
        height=480,
        data_types=["rgb"],
        update_period=0,  # every sim step
    )


@configclass
class ActionsCfg:
    """Action specifications for the Stretch."""

    # 8D. arm_action (delta)
    arm_action = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_lift", "joint_arm_l.*", "joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"],
        use_zero_offset=True,
    )

    # 3D. base_action (delta)
    base_action = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=BASE_JOINT_NAMES,
        use_zero_offset=True,
    )

    # 2D. gripper_action (delta)
    gripper_action = isaac_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_gripper_.*"],
        use_zero_offset=True,
    )

    # Total Action Dimension = 13 (head controlled separately via set_joint_position_target)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):

        # 8D. The relative joint positions (current minus default) for the 8 arm/wrist joints. (joint space)
        arm_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            },
        )

        # 3D. The relative joint positions of the base (x, y, rotation). (joint space)
        base_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=BASE_JOINT_NAMES),
            },
        )

        # 11D. The relative joint velocities for the arm, wrist, and base joints. (joint space)
        arm_joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES + BASE_JOINT_NAMES,
                ),
            },
        )

        # 3D. The vextor from the gripper to the drawer handle. (world frame)
        handle_rel = ObsTerm(
            func=handle_rel_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"]),
                "target_cfg": SceneEntityCfg(
                    "cabinet", body_names=["drawer_handle_top"]
                ),
            },
        )

        # 2D. Relative joint positions of the gripper fingers. (joint space)
        gripper_state = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_gripper_.*"]),
            },
        )

        # 1D. The absolute position of the drawer drawer_top_joint (joint space)
        drawer_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "cabinet", joint_names=["drawer_top_joint"]
                ),
            },
        )

        # Total Observation Dimention = 28D

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTermCfg(
        func=isaac_mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the Cabinet Opening task."""

    door_opening = RewTerm(
        func=reward_cabinet_opening_proportional,
        weight=50.0,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "max_open": 0.4,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for ending the episode."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class StretchEnvCfg(ManagerBasedRLEnvCfg):
    scene: StretchSceneCfg = StretchSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 3600.0
        self.viewer.eye = (4.0, 2.0, 3.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
