from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import mdp as isaac_mdp
from config.stretch_cfg import STRETCH_CFG


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


@configclass
class ActionsCfg:
    """Action specifications for the Stretch."""

    # 8D. arm_action
    arm_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_lift",
            "joint_arm_l.*",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
        ],
    )

    # 3D. base_action
    base_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_x", "joint_y", "joint_rot_z"],
    )

    # 2D. gripper_action
    gripper_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_gripper_.*"],
        use_default_offset=False,
    )

    # Total Action Dimention = 13


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):

        # 8D. The relative joint positions (current minus default) for the 8 arm/wrist joints. (joint space)
        arm_joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_lift",
                        "joint_arm_l0",
                        "joint_arm_l1",
                        "joint_arm_l2",
                        "joint_arm_l3",
                        "joint_wrist_yaw",
                        "joint_wrist_pitch",
                        "joint_wrist_roll",
                    ],
                ),
            },
        )

        # 3D. The relative joint positions of the base (x, y, rotation). (joint space)
        base_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["joint_x", "joint_y", "joint_rot_z"]
                ),
            },
        )

        # 11D. The relative joint velocities for the arm, wrist, and base joints. (joint space)
        arm_joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_lift",
                        "joint_arm_l0",
                        "joint_arm_l1",
                        "joint_arm_l2",
                        "joint_arm_l3",
                        "joint_wrist_yaw",
                        "joint_wrist_pitch",
                        "joint_wrist_roll",
                        "joint_x",
                        "joint_y",
                        "joint_rot_z",
                    ],
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
