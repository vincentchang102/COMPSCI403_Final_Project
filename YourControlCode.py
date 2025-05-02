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
    self.boxCtrlhdl.set_difficulty(0.25) #set difficulty level, base = 0.25

    self.insertion_started = False 
    self.insert_start_pos = None

  def update(self):
    # Baseline implementation: completion time 6.25
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

    if np.abs(np.dot(unit_normal, unit_vector)) > 0.95:
      # Aligned? 
      print("axis aligned")
      self.insertion_started = True
      self.insert_start_pos = ee_pos_coord
    
    # FIXME 
    if self.insertion_started:
        print(f"Do something")


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
    # target_jpos = initial_jpos + 0.01 * np.linalg.pinv(J_pose) @ pose_err
    # Speeding up convergence 
    target_jpos = initial_jpos + np.linalg.pinv(J_pose) @ pose_err

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
    
    
