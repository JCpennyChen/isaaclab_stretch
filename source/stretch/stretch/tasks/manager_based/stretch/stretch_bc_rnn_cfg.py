from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import mdp as isaac_mdp
from config.stretch_cfg import STRETCH_CFG


def reward_cabinet_opening_proportional(
    env, cabinet_cfg: SceneEntityCfg, max_open: float = 0.4
):
    """
    Reward that scales non-linearly with how open the drawer is.
    """
    door_pos = env.scene[cabinet_cfg.name].data.joint_pos[:, cabinet_cfg.joint_ids]
    normalized_pos = torch.clamp(door_pos / max_open, min=0.0, max=1.0)
    return torch.sum(torch.pow(normalized_pos, 2), dim=-1)


def position_rel(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """
    Computes the relative position of a target body w.r.t to an asset body.
    Returns: (target_pos - asset_pos)
    """
    asset_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids[0]]
    target_pos = env.scene[target_cfg.name].data.body_pos_w[:, target_cfg.body_ids[0]]
    return target_pos - asset_pos


def randomize_head_look_at_cabinet(env, env_ids, asset_cfg, target_cfg):
    """
    Calculates the pan and tilt angles required for the Stretch robot
    to look at the target cabinet and applies them to the joints.
    """
    robot_root = env.scene[asset_cfg.name].data.root_state_w[env_ids, :3]
    target_root = env.scene[target_cfg.name].data.root_state_w[env_ids, :3]

    head_height = 1.1
    rel_pos = target_root - robot_root
    rel_pos[:, 2] -= head_height

    pan = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])

    dist_xy = torch.norm(rel_pos[:, :2], dim=1)
    tilt = -torch.atan2(rel_pos[:, 2], dist_xy)

    robot = env.scene[asset_cfg.name]
    pan_idx = robot.find_joints("joint_head_pan")[0]
    tilt_idx = robot.find_joints("joint_head_tilt")[0]

    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_pos[:, pan_idx] = pan
    joint_pos[:, tilt_idx] = tilt

    robot.write_joint_state_to_sim(
        joint_pos, robot.data.joint_vel[env_ids], env_ids=env_ids
    )


def compute_head_tracking_actions(robot_base_pos, target_pos):
    head_height = 1.1

    rel_pos = target_pos - robot_base_pos
    rel_pos[:, 2] -= head_height
    pan = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
    dist_xy = torch.norm(rel_pos[:, :2], dim=1)
    tilt = torch.atan2(rel_pos[:, 2], dist_xy)

    return torch.stack([pan, tilt], dim=-1)


@configclass
class StretchSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with the Stretch robot."""

    bg_env = AssetBaseCfg(
        prim_path="/World/Env",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/johnchen/SharedSSD/JohnChen/stretch/assets/ground_plane/default_environment.usd"
        ),
    )

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
                velocity_limit=100.0,
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    robot = STRETCH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_color_frame/camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, -0.707, 0.707, 0.0),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the Stretch."""

    body_joints = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_x",
            "joint_y",
            "joint_rot_z",
            "joint_lift",
            "joint_wrist_.*",
            "joint_gripper_.*",
            "joint_arm_.*",
            "joint_head_.*",
        ],
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        cabinet_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        eef_pose = ObsTerm(
            func=isaac_mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"])
            },
        )

        handle_rel = ObsTerm(
            func=position_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"]),
                "target_cfg": SceneEntityCfg(
                    "cabinet", body_names=["drawer_handle_top"]
                ),
            },
        )

        gripper_state = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_gripper_.*"]),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class RGBCfg(ObsGroup):
        camera_image = ObsTerm(
            func=isaac_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_camera"),
                "data_type": "rgb",
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb: RGBCfg = RGBCfg()


@configclass
class EventCfg:
    reset_all = EventTermCfg(
        func=isaac_mdp.reset_scene_to_default,
        mode="reset",
    )

    reset_robot_base = EventTermCfg(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTermCfg(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!joint_y$).*"]),
        },
    )

    look_at_cabinet = EventTermCfg(
        func=randomize_head_look_at_cabinet,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("cabinet"),
        },
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
