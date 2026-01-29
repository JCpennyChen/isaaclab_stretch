import yaml
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import UrdfFileCfg
from isaaclab.sim.schemas import JointDrivePropertiesCfg

YAML_PATH = "/home/johnchen/SharedSSD/JohnChen/stretch/assets/configs/robot_configs/stretch_joint.yml"

# --- 2. LOAD THE YAML DATA ---
print(f"Loading robot config from: {YAML_PATH}")
with open(YAML_PATH, "r") as file:
    yaml_content = yaml.safe_load(file)

curobo_cfg = yaml_content["robot_cfg"]

# Get the absolute path we set in the YAML
urdf_path_from_yaml = curobo_cfg["kinematics"]["urdf_path"]

# Extract Default Joint Positions
curobo_joint_names = curobo_cfg["kinematics"]["cspace"]["joint_names"]
curobo_joint_values = curobo_cfg["kinematics"]["cspace"]["retract_config"]
default_joint_map = dict(zip(curobo_joint_names, curobo_joint_values))

default_joint_drive = JointDrivePropertiesCfg(
    drive_type="force",
    stiffness=1050.0,
    damping=55.0,
)

default_joint_drive.target_type = "position"
default_joint_drive.gains = None

STRETCH_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=urdf_path_from_yaml,
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=default_joint_drive,
        make_instanceable=True,
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
        # BASE
        "ghost_base": ImplicitActuatorCfg(
            joint_names_expr=["rotate_z", "base_forward"],
            effort_limit=20000.0,
            stiffness=1050.0,
            damping=55.0,
            velocity_limit=100.0,
        ),
        # LIFT
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            stiffness=1050.0,
            damping=55.0,
            effort_limit=10000.0,
        ),
        # ARM
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_arm_l0",
                "joint_arm_l1",
                "joint_arm_l2",
                "joint_arm_l3",
            ],
            stiffness=1050.0,
            damping=55.0,
        ),
        # WRIST
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_wrist_yaw",
                "joint_wrist_pitch",
                "joint_wrist_roll",
            ],
            stiffness=1050.0,
            damping=20.0,
        ),
        # GRIPPER
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint_gripper_finger_.*"],
            effort_limit=200.0,
            stiffness=1050.0,
            damping=5.0,
        ),
        # HEAD
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*"],
            stiffness=500.0,
            damping=10.0,
        ),
    },
)
