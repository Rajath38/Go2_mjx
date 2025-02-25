from go2 import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from go2_deploy.common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from go2_deploy.common.rotation_helper import get_gravity_orientation, transform_imu_data
from go2_deploy.common.remote_controller import RemoteController, KeyMap
from go2_deploy.config import Config
from go2_deploy.common.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.utils.thread import RecurrentThread
from inter_process_com.publisher import GetSetObservations, ThreadStatus


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions)
        self.last_action = np.zeros(self.config.num_actions)


        self.get_set_obs = GetSetObservations()
        self.low_state = unitree_go_msg_dds__LowState_()

        #initialize  a subscriber
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # wait for the subscriber to receive data indicating that the go2 is connected successfully
        self.wait_for_low_state()
        

        status, result = self.msc.CheckMode()
        
        while result['name']:
            print(f"Trying to deactivate the motion control-related service..")
            self.sc.StandDown()
            code, _ = self.msc.ReleaseMode()
            if (code == 0):
                print("ReleaseMode succeeded.")
            else:
                print("ReleaseMode failed. Error code: ")
            status, result = self.msc.CheckMode()
            time.sleep(1)


        #init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.run, name="writebasiccmd"
        )
        print("Enter START to run main controller")
        while self.remote_controller.button[KeyMap.Y] != 1:
            pass 

        self.lowCmdWriteThreadPtr.Start()

    def run(self):
        
        # Get the current joint position and velocity
        for i in self.config.leg_joint2motor_idx:
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)


        
        # create observation
        gravity_orientation = get_gravity_orientation(quat) # joint velocities
        qj_obs = self.qj.copy()  # joint positions
        dqj_obs = self.dqj.copy()  # joint velocities

        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale  #joint offsets
        dqj_obs = dqj_obs * self.config.dof_vel_scale 

        ang_vel = ang_vel * self.config.ang_vel_scale

        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        """linvel, #3
        gyro, #3
        gravity, #3
        del_joint_angles, #12
        joint_velocities, #12
        self._last_action, #12
        self.PJ.get()['XYyaw'], #3"""

        num_actions = self.config.num_actions
        self.obs[3:6] = ang_vel
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions*2] = dqj_obs
        self.obs[9 + num_actions*2 : 9 + num_actions*3] = self.last_action
        self.obs[9 + num_actions*3: 9 + num_actions*3 + 3] = self.cmd 

        self.last_action = self.action



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{args.config}"
   
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()
    
    controller.start()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.X] == 1:
                break
        except KeyboardInterrupt:
            break
   
    print("Exit")
