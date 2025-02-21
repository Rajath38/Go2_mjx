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


class Controller:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # go2 uses the go msg type
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        #initialize a Publisher
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        #initialize  a subscriber
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # wait for the subscriber to receive data indicating that the go2 is connected successfully
        self.wait_for_low_state()
        
        #Routine to switch off Sport mode to void confusion 
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

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
        
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Enter default robot pose.")
        print("Waiting for the start signal...")

        while self.remote_controller.button[KeyMap.F1] != 1:
            print("Moving to default pos.")
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

            # move to default pos
            for i in range(num_step):
                alpha = i / num_step
                #for j in range(dof_size): #comment to only move calf_FL
                j = 2
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                self.send_cmd(self.low_cmd)
                time.sleep(self.config.control_dt)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.run, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def run(self):
        
        self.motiontime += 1
        
        if self.motiontime < 20:
            self.qInit[0] = self.low_state.motor_state[0].q
            self.qInit[1] = self.low_state.motor_state[1].q
            self.qInit[2] = self.low_state.motor_state[2].q
        
        if 10 <= self.motiontime < 400:
            self.rate_count += 1
            rate = self.rate_count / 200.0
            self.Kp = [2.0] * 3
            self.Kd = [0.5] * 3
            
            for i in range(3):
                self.qDes[i] = self.joint_linear_interpolation(self.qInit[i], self.sin_mid_q[i], rate)
        
        if self.motiontime >= 400:
            self.sin_count += 1
            t = self.dt * self.sin_count
            freq_Hz = 1.0
            freq_rad = freq_Hz * 2 * np.pi
            sin_joint1 = 0.6 * np.sin(t * freq_rad)
            sin_joint2 = -0.9 * np.sin(t * freq_rad)
            
            self.qDes[0] = self.sin_mid_q[0]
            self.qDes[1] = self.sin_mid_q[1] + sin_joint1
            self.qDes[2] = self.sin_mid_q[2] + sin_joint2
        
        self.low_cmd.motor_cmd[2].q = self.qDes[2]
        self.low_cmd.motor_cmd[2].dq = 0
        self.low_cmd.motor_cmd[2].kp = self.Kp[2]
        self.low_cmd.motor_cmd[2].kd = self.Kd[2]
        self.low_cmd.motor_cmd[2].tau = 0

        print(f"lowcmd: {self.low_cmd.motor_cmd}")
        print("---------------------------------------------")  #only issue is lowcmd_does not change from time to time?

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{args.config}"
   
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    input("Initialization complete. Press Enter to start the motion...")
    
    controller.Start()

    while True:        
            time.sleep(25)

    """ Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")"""
