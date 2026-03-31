import argparse
import sys
import os
from isaaclab.app import AppLauncher

# 1. Launch App
parser = argparse.ArgumentParser(description="Full Body Teleop")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input
import omni.appwindow
import torch
from isaaclab.envs import ManagerBasedRLEnv

# Path injection
task_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(task_dir)
sys.path.append(os.path.join(task_dir, "config"))

from stretch_env_cfg import StretchEnvCfg


class FullController:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()

    def is_pressed(self, key_name):
        key_enum = getattr(carb.input.KeyboardInput, key_name)
        return self._input.get_keyboard_value(self._keyboard, key_enum) > 0.5


def main():
    env_cfg = StretchEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    controller = FullController()

    # --- TUNING SPEEDS ---
    BASE_SPEED = 2.0
    LIFT_SPEED = 3.0
    GRAVITY_LIFT_COMP = 0.05
    GRVITY_WRIST_PITCH_COMP = 0.03
    ARM_SPEED = 0.2
    WRIST_SPEED = 2.0
    GRIPPER_CLOSE_POS = -0.1
    GRIPPER_OPEN_POS = 0.4
    gripper_goal = GRIPPER_CLOSE_POS

    while simulation_app.is_running():
        with torch.inference_mode():
            # Base (I/K/J/L)
            vel_fwd = 0.0
            vel_turn = 0.0

            if controller.is_pressed("I"):
                vel_fwd = BASE_SPEED
            if controller.is_pressed("K"):
                vel_fwd = -BASE_SPEED
            if controller.is_pressed("J"):
                vel_turn = BASE_SPEED
            if controller.is_pressed("L"):
                vel_turn = -BASE_SPEED

            # Lift (C/V)
            vel_lift = 0.0
            if controller.is_pressed("C"):
                vel_lift = LIFT_SPEED
            elif controller.is_pressed("V"):
                vel_lift = -LIFT_SPEED
            else:
                vel_lift = GRAVITY_LIFT_COMP

            # Arm (Z/X)
            vel_arm = 0.0
            if controller.is_pressed("Z"):
                vel_arm += ARM_SPEED
            if controller.is_pressed("X"):
                vel_arm -= ARM_SPEED

            # --- WRIST CONTROLS ---

            # Pitch (N / M) -> Index 8 (NEEDS GRAVITY HOLD)
            vel_pitch = 0.0
            if controller.is_pressed("N"):
                vel_pitch += WRIST_SPEED
            elif controller.is_pressed("M"):
                vel_pitch -= WRIST_SPEED
            else:
                vel_pitch = GRVITY_WRIST_PITCH_COMP

            # Yaw (, / .) -> Index 7
            vel_yaw = 0.0
            if controller.is_pressed("COMMA"):
                vel_yaw += WRIST_SPEED
            if controller.is_pressed("PERIOD"):
                vel_yaw -= WRIST_SPEED

            # Roll (; / ') -> Index 9
            vel_roll = 0.0
            if controller.is_pressed("SEMICOLON"):
                vel_roll += WRIST_SPEED
            if controller.is_pressed("APOSTROPHE"):
                vel_roll -= WRIST_SPEED

            # --- GRIPPER LOGIC (With Debugging) ---
            if controller.is_pressed("LEFT_BRACKET"):
                gripper_goal = GRIPPER_OPEN_POS

            if controller.is_pressed("RIGHT_BRACKET"):
                gripper_goal = GRIPPER_CLOSE_POS

            # --- Construct Actions ---
            actions = torch.zeros((env.num_envs, 12), device=env.device)

            actions[:, 0] = vel_turn
            actions[:, 1] = vel_fwd
            actions[:, 2] = vel_lift
            actions[:, 3:7] = vel_arm
            actions[:, 7] = vel_yaw
            actions[:, 8] = vel_pitch
            actions[:, 9] = vel_roll
            actions[:, 10] = gripper_goal
            actions[:, 11] = gripper_goal

            env.step(actions)
            env.render()

    env.close()


if __name__ == "__main__":
    main()
