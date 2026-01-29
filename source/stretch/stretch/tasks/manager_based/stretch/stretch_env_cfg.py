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
import isaaclab.sim as sim_utils
import torch


from isaaclab.envs import mdp as isaac_mdp
from config.stretch_cfg import STRETCH_CFG


def reward_distance_to_handle(
    env, robot_cfg: SceneEntityCfg, cabinet_cfg: SceneEntityCfg
):
    """
    Reward for minimizing distance between the robot's grasp center and the cabinet handle.
    """
    gripper_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids]
    handle_pos = env.scene[cabinet_cfg.name].data.body_pos_w[:, cabinet_cfg.body_ids]
    distance = torch.norm(gripper_pos - handle_pos, dim=-1)
    return 1.0 / (1.0 + distance.squeeze(-1) ** 2)


def reward_cabinet_opened(env, cabinet_cfg: SceneEntityCfg):
    """
    Reward for opening the cabinet door.
    """
    door_pos = env.scene[cabinet_cfg.name].data.joint_pos[:, cabinet_cfg.joint_ids]
    return torch.sum(door_pos, dim=-1)


def terminate_if_door_open(env, asset_cfg: SceneEntityCfg, threshold: float = 1.0):
    """
    Terminate the episode if the cabinet door is opened beyond 'threshold' radians.
    """
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.any(joint_pos > threshold, dim=-1)


@configclass
class StretchSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with the Stretch robot."""

    # Ground Plane
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
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )

    # Obstacle Cube
    obstacle_cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            # Add these to enable physics/gravity:
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
    )

    # Robot
    robot = STRETCH_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


@configclass
class ActionsCfg:
    """Action specifications for the Stretch."""

    # 1. BASE
    base = isaac_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rotate_z", "base_forward"],
        scale=1.0,
        use_default_offset=False,
    )
    # 2. LIFT
    lift_velocity = isaac_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_lift"],
        scale=1.0,
    )
    # 3. ARM
    arm_velocity = isaac_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_arm_l.*"],
        scale=1.0,
    )
    # 4. WRIST (
    wrist_velocity = isaac_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"],
        scale=1.0,
    )
    # 5. GRIPPER
    gripper = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_gripper_finger_.*"],
        scale=1.0,
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

        # Cabinet Base Position (XYZ)
        cabinet_base_pos = ObsTerm(
            func=isaac_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # Robot Base Position (XYZ)
        robot_root_pos = ObsTerm(
            func=isaac_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # Robot Base Rotation (Quaternion)
        robot_root_rot = ObsTerm(
            func=isaac_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # Gripper Center Pose (Position + Rotation)
        eef_pose = ObsTerm(
            func=isaac_mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"])
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

    # 2. Reaching the Handle
    reach_handle = RewTerm(
        func=reward_distance_to_handle,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["link_grasp_center"]),
            "cabinet_cfg": SceneEntityCfg("cabinet", body_names=["drawer_handle_top"]),
        },
    )

    # 3. Opening the Door
    door_opening = RewTerm(
        func=reward_cabinet_opened,
        weight=10.0,
        params={
            # Make sure you have 'joint_names' here!
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
        },
    )

    # 4. Penalties (Penalize large, jerky actions to keep movement smooth)
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)

    # 5. Penalize high joint velocities
    joint_vel = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for ending the episode."""

    # 1. Time Out
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    # 2. Success Condition

    door_is_open = DoneTerm(
        func=terminate_if_door_open,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["door_left_joint"]),
            "threshold": 1.0,
        },
    )


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
