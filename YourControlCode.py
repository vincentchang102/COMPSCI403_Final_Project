import mujoco
import numpy as np
import mujoco
from BoxControlHandler import BoxControlHandle

class YourCtrl:
  
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    self.boxCtrlhdl = BoxControlHandle(self.m,self.d)
    self.boxCtrlhdl.set_difficulty(0.3) #set difficulty level, base = 0.25

    self.insertion_started = False 
    self.insert_start_pos = None

  def update(self):
    # Get the current position of the box
    box_sensor1_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor1")
    box_sensor2_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor2")
    box_sensor3_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor3")
    box_sensor4_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor4") 

    boxmould_pos1 = self.d.sensordata[box_sensor1_idx*3:box_sensor1_idx*3+3]
    boxmould_pos2 = self.d.sensordata[box_sensor2_idx*3:box_sensor2_idx*3+3]
    boxmould_pos3 = self.d.sensordata[box_sensor3_idx*3:box_sensor3_idx*3+3]
    boxmould_pos4 = self.d.sensordata[box_sensor4_idx*3:box_sensor4_idx*3+3]
    
    box_ori,normal = self.boxCtrlhdl.box_orientation(boxmould_pos1, boxmould_pos2,boxmould_pos3,boxmould_pos4)
    target_ori = self.boxCtrlhdl.rotate_quat_90_y(box_ori)


    # Calculate if end effector is aligned with normal vector of the box 
    ee_pos_coord = self.boxCtrlhdl._get_ee_position()
    box_mdpt = self.boxCtrlhdl.box_midpoint(boxmould_pos1, boxmould_pos2,boxmould_pos3,boxmould_pos4)
    vector = box_mdpt - ee_pos_coord
    unit_vector = vector / np.linalg.norm(vector)
    unit_normal = normal / np.linalg.norm(normal)

    # Check if the end effector is aligned with the normal vector of the box
    if np.abs(np.dot(unit_normal, unit_vector)) > 0.95:
      print("axis aligned")
      self.insertion_started = True
      self.insert_start_pos = ee_pos_coord
    
    if self.insertion_started:
      ########### Use Operational Space Control to move the end effector towards the box ###########

      



      # 2. Get EE pose error
      # EE_Ori = self.boxCtrlhdl._get_ee_orientation()
      # pos_err = self.boxCtrlhdl.get_EE_pos_err()
      # # Compute the orientation error as a quaternion
      # quat_err = self.boxCtrlhdl.quat_multiply(target_ori, self.boxCtrlhdl.quat_inv(EE_Ori))
      # ori_err_quat = self.boxCtrlhdl.quat2so3(quat_err)
      # # Combine position and orientation errors into a single pose error
      # ori_err = ori_err_quat
      # pose_err = np.concatenate((pos_err, ori_err))

      # 3. Compute the Jacobian
      nv = self.m.nv  # Number of degrees of freedom
      jacp = np.zeros((3, nv))  # Jacobian for position
      jacr = np.zeros((3, nv))  # Jacobian for orientation
      mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.boxCtrlhdl.ee_id)
      # Combine the position and orientation Jacobians
      J_pose = np.concatenate((jacp[:, :6], jacr[:, :6]))

      # 4. Compute Mass Matrix
      A = np.zeros((nv, nv))
      mujoco.mj_fullM(self.m, A, self.d.qM)
      ArmMassMtx = A[:6, :6]
      
      # 5. Compute Task Space Inertia Matrix
      Minv = np.linalg.pinv(ArmMassMtx)
      Lambda = np.linalg.pinv(J_pose @ Minv @ J_pose.T)

      # 6. Compute End effector velocity
      ee_velocity = J_pose @ self.d.qvel[:6]

      # 7. Operational Space Pd Control
      Kp = np.diag([300, 300, 300, 100, 100, 100])
      Kd = np.diag([14, 14, 14, 10, 10, 10])

      # Calculate velocity error
      desired_direction = -unit_normal # DIRECTION OF MOVEMENT
      desired_velocity = desired_direction * 5.0  # ADJUST THIS CONSTANT TO CHANGE STRENGTH OF MOVEMENT
      linear_velocity_error = desired_velocity - ee_velocity[:3]  # Only linear part
      angular_velocity_error = -ee_velocity[3:]  # No desired rotation

      vel_error = np.concatenate((linear_velocity_error, angular_velocity_error))

      # Calculate F_task
      F_task = Kp @ vel_error - Kd @ ee_velocity

      # 8. Compute the control force
      tau = J_pose.T @ (Lambda @ F_task) + self.d.qfrc_bias[:6]
      self.d.ctrl[:6] = tau

    else:
      ########### Align the end effector with the box ###########

      nv = self.m.nv
      jacp = np.zeros((3, nv))
      jacr = np.zeros((3, nv))

      EE_Ori = self.boxCtrlhdl._get_ee_orientation()

      pos_err = self.boxCtrlhdl.get_EE_pos_err()

      mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.boxCtrlhdl.ee_id)
      quat_err = self.boxCtrlhdl.quat_multiply(target_ori, self.boxCtrlhdl.quat_inv(EE_Ori))
      ori_err_quat = self.boxCtrlhdl.quat2so3(quat_err)

      ori_err = ori_err_quat
      pose_err = np.concatenate((pos_err, ori_err))

      J_pose = np.concatenate((jacp[:, :6], jacr[:,:6]))
      
      initial_jpos = np.copy(self.d.qpos[:6])
      target_jpos = initial_jpos + 1 * np.linalg.pinv(J_pose) @ pose_err

      self.d.qpos[:6] = target_jpos
      mujoco.mj_kinematics(self.m, self.d)

      velocity = self.d.qvel[:6]
      jpos_error = target_jpos - initial_jpos
      
    
      A = np.zeros((nv,nv))
      mujoco.mj_fullM(self.m, A, self.d.qM)
      ArmMassMtx = A[:6,:6]
      kp = 150
      kd = 10
      control_signal = ArmMassMtx @ (kp * jpos_error - kd * velocity) + self.d.qfrc_bias[:6]

      self.d.ctrl[:6] = control_signal
