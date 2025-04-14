import mujoco
import numpy as np
from scipy.linalg import inv, eig

class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 50.0
    self.kd = 3.0


  def CtrlUpdate(self):
    for i in range(6):
       self.d.ctrl[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]
    return True 



