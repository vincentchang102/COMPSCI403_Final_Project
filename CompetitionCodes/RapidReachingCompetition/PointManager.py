import mujoco
import numpy as np
import time
# import ReachingCtrlExample
import YourControlCode

class PointHandle:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d

        # Points tracking
        self.points_active = []
        self.current_target_idx = 0

        # Get end effector body ID
        self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")


    def initialize_points(self, num_points: int):
        """Initialize tracking for randomly generated points"""
        self.num_points = num_points
        self.points_active = [True] * self.num_points
        print(f"Initialized {num_points} points for tracking")
        # Reset timing metrics
        self.start_time = time.time()
        self.point_start_times = {0: self.start_time}  # Start timing for first point
        self.point_completion_times = {}
        self.total_time = None

        self.target_points = np.zeros((3, num_points)) 

        for i in range(num_points):  # Corrected loop
            site_id = self.m.site(f'point{i}').id  # Ensure this correctly retrieves an ID
            self.target_points[:, i] = self.d.site_xpos[site_id].copy()  # Corrected indexing

        # self.reaching_ctrl = ReachingCtrlExample.ReachingCtrl(self.m, self.d, self.target_points)
        self.reaching_ctrl = YourControlCode.YourCtrl(self.m, self.d, self.target_points)

    def _get_ee_position(self):
        """Get current end effector position"""
        return self.d.xpos[self.ee_id].copy()


    def _check_point_reached(self, threshold=0.01):
        """Check if current target point is reached"""
        ee_pos = self._get_ee_position()
        for i in range(self.num_points):
            if(self.points_active[i] == True):
                distance = np.linalg.norm(ee_pos - self.target_points[:,i])
                if distance < threshold:
                    # Mark point as reached
                    self.points_active[i] = False

                    # Change point color to green in visualization
                    site_id = self.m.site(f'point{i}').id
                    self.m.site_rgba[site_id] = np.array([0.0, 1.0, 0.0, 1.0])  # Green

                    print(f"Point {self.current_target_idx} reached! Distance: {distance:.3f}")
                    print(f"Points remaining: {sum(self.points_active)}")

        if not any(self.points_active):
            print("All points reached!")
            self.total_time = time.time() - self.start_time
            print("\nFinal Timing Results:")
            print("=====================")
            print(f"Total time: {self.total_time:.2f} seconds")
            exit()
            return False

        else:
            return True

    def update(self):
        """Update controller state and compute control signals"""
        # Apply control signal
        self.d.ctrl = self.reaching_ctrl.CtrlUpdate()

        # Check if we need to move to next point
        if not self._check_point_reached():
            self.d.ctrl[:6] = 0  # Stop the robot
            return

