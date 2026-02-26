from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp import JointPositionAction
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import mdp as isaac_mdp
from config.stretch_cfg import STRETCH_CFG


class ReplicatedJointPositionAction(JointPositionAction):
    """
    Takes a 1-dimensional action and replicates it across all selected joints.
    Useful for telescoping arms or parallel grippers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_dim = 1

    @property
    def action_dim(self):
        """Force the environment to see this as 1 dimension."""
        return 1

    def process_action(self, action: torch.Tensor):
        expanded_action = action.expand(-1, self.num_joints)
        super().process_action(expanded_action)


def reward_distance_to_handle(
    env, robot_cfg: SceneEntityCfg, cabinet_cfg: SceneEntityCfg, offset: list = None
):
    """
    Reward for minimizing distance to the handle, with an optional local offset.
    """
    gripper_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids]
    handle_pos = env.scene[cabinet_cfg.name].data.body_pos_w[:, cabinet_cfg.body_ids]
    handle_quat = env.scene[cabinet_cfg.name].data.body_quat_w[:, cabinet_cfg.body_ids]

    if offset is not None:
        offset_vec = torch.tensor(offset, device=env.device).repeat(
            handle_pos.shape[0], 1
        )
        target_pos = handle_pos + quat_apply(handle_quat, offset_vec)
    else:
        target_pos = handle_pos

    distance = torch.norm(gripper_pos - target_pos, dim=-1)
    return 1.0 / (1.0 + distance.squeeze(-1) ** 2)


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
                damping=200.0,
            ),
        },
    )

    obstacle_cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
    )

    robot = STRETCH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_color_frame/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the Stretch."""

    body_joints = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "rotate_z",
            "base_forward",
            "joint_lift",
            "joint_wrist_.*",
            "joint_gripper_.*",
        ],
        use_default_offset=False,
    )

    arm_extension = isaac_mdp.JointPositionActionCfg(
        class_type=ReplicatedJointPositionAction,
        asset_name="robot",
        joint_names=["joint_arm_.*"],
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)

        # Cabinet Joint Positions (Absolute)
        cabinet_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # Gripper Center Pose (Position + Rotation)
        eef_pose = ObsTerm(
            func=isaac_mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"])
            },
        )

        # Relative position from gripper center to cabinet handle
        handle_rel = ObsTerm(
            func=position_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"]),
                "target_cfg": SceneEntityCfg(
                    "cabinet", body_names=["drawer_handle_top"]
                ),
            },
        )

        # Gripper Finger Positions
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
class EventCfg:
    reset_all = EventTermCfg(
        func=isaac_mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the Cabinet Opening task."""

    # 1. Survival
    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)

    # 2. Opening the Door
    door_opening = RewTerm(
        func=reward_cabinet_opening_proportional,
        weight=50.0,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "max_open": 0.4,
        },
    )

    # reach_handle = RewTerm(
    #     func=reward_distance_to_handle,
    #     weight=10.0,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"]),
    #         "cabinet_cfg": SceneEntityCfg("cabinet", body_names=["drawer_handle_top"]),
    #         "offset": [0.305, 0.0, 0.01],
    #     },
    # )

    # Penalize large, jerky actions to keep movement smooth
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)

    # Penalize high joint velocities
    joint_vel = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
