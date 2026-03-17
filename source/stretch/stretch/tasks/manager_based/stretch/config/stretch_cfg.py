import yaml
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import UsdFileCfg

# =========================================================
# 1. SETUP PATHS
# =========================================================
PROJECT_ROOT = Path("/home/johnchen/SharedSSD/JohnChen/stretch")
YAML_PATH = PROJECT_ROOT / "assets/robot_configs/stretch_fake_joint.yml"
curobo_cfg = yaml.safe_load(YAML_PATH.read_text())["robot_cfg"]
raw_path = curobo_cfg["kinematics"]["usd_path"].replace("assets/", "", 1)
usd_full_path = str((PROJECT_ROOT / "assets" / raw_path).resolve(strict=True))

# =========================================================
# HOME POSITION
# =========================================================
curobo_joint_names = curobo_cfg["kinematics"]["cspace"]["joint_names"]
curobo_joint_values = curobo_cfg["kinematics"]["cspace"]["retract_config"]
default_joint_map = dict(zip(curobo_joint_names, curobo_joint_values))

# =========================================================
# DEFINE THE ROBOT CONFIG
# =========================================================
STRETCH_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=usd_full_path,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=1e-3, rest_offset=0.0
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=default_joint_map,
    ),
    actuators={
        # Base Joints
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y", "joint_rot_z"],
            stiffness=4000.0,
            damping=400.0,
            effort_limit=400.0,
            velocity_limit=1.0,
        ),
        # Lift, Arm, & Wrist Joints
        "arm_and_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift", "joint_arm_l.*", "joint_wrist_.*"],
            stiffness=4000.0,
            damping=400.0,
            effort_limit=200.0,
            velocity_limit=1.0,
        ),
        # Gripper Joints
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_.*"],
            stiffness=2000.0,
            damping=200.0,
            effort_limit=100.0,
            velocity_limit=0.5,
        ),
        # Passive Joints
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*", "joint_.*_wheel"],
            stiffness=4000.0,
            damping=400.0,
        ),
    },
)
