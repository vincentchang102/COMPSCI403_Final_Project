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
    self.boxCtrlhdl.set_difficulty(0.6) #set difficulty level

   

  def update(self):
    box_sensor1_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor1")
    box_sensor2_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor2")
    box_sensor3_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor3")
    box_sensor4_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor4") 

    boxmould_pos1 = self.d.sensordata[box_sensor1_idx*3:box_sensor1_idx*3+3]
    boxmould_pos2 = self.d.sensordata[box_sensor2_idx*3:box_sensor2_idx*3+3]
    boxmould_pos3 = self.d.sensordata[box_sensor3_idx*3:box_sensor3_idx*3+3]
    boxmould_pos4 = self.d.sensordata[box_sensor4_idx*3:box_sensor4_idx*3+3]

    box_ori,_ = self.boxCtrlhdl.box_orientation(boxmould_pos1, boxmould_pos2,boxmould_pos3,boxmould_pos4)
    target_ori = self.boxCtrlhdl.rotate_quat_90_y(box_ori)
  
    # 2. Get EE pose error
    EE_Ori = self.boxCtrlhdl._get_ee_orientation()
    pos_err = self.boxCtrlhdl.get_EE_pos_err()
    # Compute the orientation error as a quaternion
    quat_err = self.boxCtrlhdl.quat_multiply(target_ori, self.boxCtrlhdl.quat_inv(EE_Ori))
    ori_err_quat = self.boxCtrlhdl.quat2so3(quat_err)
    # Combine position and orientation errors into a single pose error
    ori_err = ori_err_quat
    pose_err = np.concatenate((pos_err, ori_err))


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

    # Kp = np.diag([150, 150, 150, 50, 50, 50])
    # Kd = np.diag([10, 10, 10, 5, 5, 5])



    F_task = Kp @ pose_err - Kd @ ee_velocity

    # 8. Compute the control force
    tau = J_pose.T @ (Lambda @ F_task) + self.d.qfrc_bias[:6]
    self.d.ctrl[:6] = tau


