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

    self.qpos_error_history = np.zeros(3*12)
    self.motor_targets = np.zeros(12)


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
        linvel, #3
        gyro, #3
        gravity, #3
        del_joint_angles, #12
        #self.qpos_error_history, #36
        #feet_pos, #12
        joint_velocities, #12
        self._last_action, #12
        #self._last_last_action,
        self.PJ.get()['XYyaw'], #3
    ])
    print(f"obs:{obs}")
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    #if self._counter % self._n_substeps == 0:
    obs = self.get_obs(model, data)
    onnx_input = {"state": obs.reshape(1, -1)}
    
  
    #print(f"Height: {data.qpos[root_adr+2]}")
    onnx_pred = self._policy.run(None, onnx_input)[0][0]
    print(f"onnx_pred = {onnx_pred}")
    self._last_action = onnx_pred.copy()
    self._last_last_action = self._last_action
    self.motor_targets = onnx_pred*self._action_scale + self._default_angles 
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
      policy_path=("utils/outputs/go2_policy-5.onnx"),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      n_substeps=n_substeps,
      action_scale=0.3,
      vel_scale_x=1.5,
      vel_scale_y=0.8,
      vel_scale_rot= 2 * np.pi,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
