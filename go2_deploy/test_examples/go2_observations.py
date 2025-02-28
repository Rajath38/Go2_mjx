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
from inter_process_com.publisher import GetSetObservations 


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.get_set_obs = GetSetObservations()

        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # go2 uses the go msg type
        #self.low_cmd = unitree_go_msg_dds__LowCmd_()
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
        self.action = np.zeros(self.config.num_actions)
        self.last_action = np.zeros(self.config.num_actions)
        self.last_last_action = np.zeros(self.config.num_actions)

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
        self.remote_controller.set(self.low_state.wireless_remote)

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

    def zero_torque_state(self):
        print("For zero torque state.")
        print("Enter 'A'...to proceed")
        while self.remote_controller.button[KeyMap.A] != 1:
            pass

        print("Zero torque")
        time.sleep(self.config.control_dt)
    
    def move_to_default_pos(self):
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.kps 
        kds = self.config.kds 
        default_pos = self.config.default_angles
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        print("Press 'B' to DEFAULT ROBOT POSE...")

        while self.remote_controller.button[KeyMap.B] != 1:
            pass
            
        # move to default pos
        print(f"Moving to default robot pose")

        for i in range(num_step):
            print(f"Moving step {i}")
            alpha = i / num_step
            for j in range(dof_size): #comment to only move calf_FL
                time.sleep(self.config.control_dt)


    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in self.config.leg_joint2motor_idx:
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)


        """ noisy_linvel,  #  Typically ang vel is used, since lin velovity is not accurate estimated
        noisy_gyro,  # 3 
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["command"],  # 3"""

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



        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel   #lin_vel is yet to be added soon form est_ data
        self.obs[3:6] = ang_vel
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions*2] = dqj_obs
        self.obs[9 + num_actions*2 : 9 + num_actions*3] = self.action
        self.obs[9 + num_actions*3: 9 + num_actions*3 + 3] = self.cmd 

        #print(f"OBS: {self.obs}")
        print(f"Button A: {self.remote_controller.button[KeyMap.A]}")
        print(f"Button B: {self.remote_controller.button[KeyMap.B]}")
        print(f"Button START: {self.remote_controller.button[KeyMap.start]}")


        """# Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = 0#self.policy(obs_tensor).detach().numpy().squeeze()
        self.last_action = self.action
        self.last_last_action = self.last_action
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale"""

        time.sleep(self.config.control_dt)


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
