import torch
import numpy as np


class PIDPathTracker:
    def __init__(self, dt=0.05, linear_k=[1.5, 0.0, 0.1], angular_k=[4.0, 0.0, 0.5]):
        """
        PID Controller for Differential Drive Robot.
        linear_k = [Kp, Ki, Kd] for linear velocity
        angular_k = [Kp, Ki, Kd] for angular velocity
        """
        self.dt = dt

        self.kp_lin, self.ki_lin, self.kd_lin = linear_k
        self.kp_ang, self.ki_ang, self.kd_ang = angular_k

        # Error terms for Integral (I) and Derivative (D)
        self.prev_lin_error = 0.0
        self.sum_lin_error = 0.0

        self.prev_ang_error = 0.0
        self.sum_ang_error = 0.0

        # Limits (Safety)
        self.max_v = 0.5  # m/s
        self.max_w = 1.0  # rad/s

    def get_next_command(self, current_pose, path_tensor):
        """
        Calculates (v, w) to follow the path.
        current_pose: [x, y, theta] (Tensor)
        path_tensor: [N, 3] (Tensor)
        Returns: next_pose [x, y, theta] (Tensor) for simulation step
        """
        # 1. Find Target Waypoint (Lookahead)
        # Convert to CPU numpy for easier math
        curr_p = current_pose.cpu().numpy()
        path = path_tensor.cpu().numpy()

        # Find closest point index
        dists = np.linalg.norm(path[:, :2] - curr_p[:2], axis=1)
        min_idx = np.argmin(dists)

        # Look ahead X steps (e.g., 5 steps or ~20cm)
        target_idx = min(min_idx + 5, len(path) - 1)
        target = path[target_idx]

        # 2. Calculate Errors
        # Linear Error: Distance to target
        dx = target[0] - curr_p[0]
        dy = target[1] - curr_p[1]
        dist_error = np.sqrt(dx**2 + dy**2)

        # Angular Error: Difference in heading
        desired_yaw = np.arctan2(dy, dx)
        current_yaw = curr_p[2]
        yaw_error = desired_yaw - current_yaw

        # Normalize yaw error to [-pi, pi]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        # 3. PID Control - Linear Velocity
        # (Standard PID formula)
        self.sum_lin_error += dist_error * self.dt
        d_lin_error = (dist_error - self.prev_lin_error) / self.dt

        v_cmd = (
            (self.kp_lin * dist_error)
            + (self.ki_lin * self.sum_lin_error)
            + (self.kd_lin * d_lin_error)
        )

        self.prev_lin_error = dist_error

        # 4. PID Control - Angular Velocity
        self.sum_ang_error += yaw_error * self.dt
        d_ang_error = (yaw_error - self.prev_ang_error) / self.dt

        w_cmd = (
            (self.kp_ang * yaw_error)
            + (self.ki_ang * self.sum_ang_error)
            + (self.kd_ang * d_ang_error)
        )

        self.prev_ang_error = yaw_error

        # 5. Non-Holonomic Constraint Handling (Smart Logic)
        # If we are facing the wrong way (> 45 deg), stop moving forward and turn!
        # This prevents "drifting" sideways.
        if abs(yaw_error) > 0.5:  # ~30 degrees
            v_cmd = 0.0

        # 6. Clamp Commands
        v_cmd = np.clip(v_cmd, -self.max_v, self.max_v)
        w_cmd = np.clip(w_cmd, -self.max_w, self.max_w)

        # 7. Integrate to get Next Pose (for simulation)
        # x_new = x + v * cos(theta) * dt
        next_x = curr_p[0] + v_cmd * np.cos(curr_p[2]) * self.dt
        next_y = curr_p[1] + v_cmd * np.sin(curr_p[2]) * self.dt
        next_th = curr_p[2] + w_cmd * self.dt

        return torch.tensor([next_x, next_y, next_th], device=current_pose.device)
