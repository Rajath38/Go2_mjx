# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
from inter_process_com import publisher as pub
import go2_spot.go2_constants as consts
import time
import os

import curses



_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class OnnxController:
  """ONNX controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.3,
      vel_scale_x: float = 1.5,
      vel_scale_y: float = 0.8,
      vel_scale_rot: float = 2 * np.pi,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._last_last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps

    self.qpos_error_history = np.zeros(2*12)
    self.motor_targets = np.zeros(12)

    self.motor_targets_max = np.ones(12)*-3.14
    self.motor_targets_min = np.ones(12)*3.14

    self.motor_targets_min_limit = np.array([-0.25436466  0.51899811 -2.11845617 -0.44738526  0.50744464 -2.14778503
                                           -0.22325524  0.74209658 -2.13979741 -0.34792199  0.68587079 -2.15971877]) * 1.1   # to addd an additional 10% allowance
    self.motor_targets_max_limit = np.array([ 0.42143986  1.23518472 -1.40191968  0.25058498  1.25762399 -1.40316312
  0.34779105  1.24745273 -1.40170221  0.25281086  1.22268566 -1.40238042]) * 1.1  # to addd an additional 10% allowance

    # Initialize publisher
    self.PJ = pub.publish_cmd()

  def get_feet_pos(self, data) -> np.ndarray:
    """Return the position of the feet relative to the trunk."""
    return np.vstack([
        data.sensor(sensor_name).data
        for sensor_name in consts.FEET_POS_SENSOR
    ])

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    del_joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]

    history = np.roll(self.qpos_error_history, 12)
    history[:12] = data.qpos[7:] - self.motor_targets  # Overwrite first 12 elements
    self.qpos_error_history = history

    feet_pos = self.get_feet_pos(data).ravel()
    

    obs = np.hstack([
        #linvel, #3
        feet_pos, #12
        gyro, #3
        gravity, #3
        del_joint_angles, #12
        joint_velocities, #12
        #self.qpos_error_history, #36
        self._last_action, #12
        #self._last_last_action,
        self.PJ.get()['XYyaw'], #3
    ])
    #print(f"obs:{obs}")
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    #if self._counter % self._n_substeps == 0:
    obs = self.get_obs(model, data)
    onnx_input = {"state": obs.reshape(1, -1)}
    
  
    #print(f"Height: {data.qpos[root_adr+2]}")
    onnx_pred = self._policy.run(None, onnx_input)[0][0]
    #print(f"onnx_pred = {onnx_pred}")
    self._last_action = onnx_pred.copy()
    self._last_last_action = self._last_action
    self.motor_targets = onnx_pred*self._action_scale + self._default_angles

    # Update min and max for each individual element
    for i in range(12):
      self.motor_targets_max[i] = max(self.motor_targets_max[i], self.motor_targets[i])
      self.motor_targets_min[i] = min(self.motor_targets_min[i], self.motor_targets[i])

    #print(f"motor_targets_min:{self.motor_targets_min}")
    #print(f"motor_targets_max:{self.motor_targets_max}")
    

    if self._counter % 10 == 0:  # Print every 100 steps
        os.system("clear")  # Clears the terminal (Use "cls" for Windows)
        print(f"Step {self._counter}:")
        print(f"motor_targets_min: {self.motor_targets_min}")
        print(f"motor_targets_max: {self.motor_targets_max}")


    data.ctrl[:] = self.motor_targets
      #print(f"ctrl {data.ctrl[:]}")

def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path("go2_spot/xmls/scene_mjx_feetonly_flat_terrain.xml")
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.004
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt
  
  policy = OnnxController(
      policy_path=("utils/outputs/go2_policy-127.onnx"),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      n_substeps=n_substeps,
      action_scale=0.4,
      vel_scale_x=1.5,
      vel_scale_y=0.8,
      vel_scale_rot= 2 * np.pi,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
