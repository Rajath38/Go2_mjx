from go2 import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.utils.crc import CRC

from go2_deploy.common.command_helper import create_zero_cmd, init_cmd_go
from go2_deploy.common.remote_controller import KeyMap
from go2_deploy.config import Config
from go2_deploy.utils.publisher import GO2STATE

import onnxruntime as rt


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
   
        self.obs = GO2STATE()

        # go2 uses the go msg type
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        #initialize a Publisher
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.action = np.zeros(self.config.num_actions)
        self.last_action = np.zeros(self.config.num_actions)
        self.last_last_action = np.zeros(self.config.num_actions)
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        policy_path=("utils/outputs/go2_policy-127.onnx")

        self.action_scale=0.4
        self._last_action = np.zeros_like(self.action, dtype=np.float32)
        self.motor_targets = np.zeros(12)
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = self.crc.Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)


    def zero_torque_state(self, ask = True):
       
        if ask == True:
            print("For zero torque state.")
            print("Enter 'A'...to proceed")
            while self.get_remote_digital()[KeyMap.A] != 1:
                time.sleep(self.config.control_dt)

        print("Zero torque")
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    def get_remote_digital(self):
        return self.obs.get_remote_data("Digital")
    
    def get_remote_analog(self):
        return self.obs.get_remote_data("Analog")
    
    def move_to_pos(self, default_pos, total_time = 2, ask = True):
        # move time 2s
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.kps 
        kds = self.config.kds 
        
        dof_size = len(dof_idx)

        leg_state = self.obs.get()
        qj_obs = leg_state["joint_positions"]
        
        # record the current pos
        init_dof_pos = np.copy(qj_obs)

        if ask == True:
            print("For Default Pose.")
            print("Enter 'B'...to proceed")
            while self.get_remote_digital()[KeyMap.B] != 1:
                time.sleep(self.config.control_dt)
            
        # move to default pos
        print(f"Moving to requested robot pose")

        # move to default pos
        for i in range(num_step):
            print(f"Moving step {i}")
            alpha = i / num_step
            for j in range(dof_size): #comment to only move calf_FL
                #j = 2
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                self.send_cmd(self.low_cmd)
                time.sleep(self.config.control_dt)


    def run(self):

        """ noisy_feet_pos, # 12 # if we remove this legs dont touch properly with floor
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["command"],  # 3"""


        # create observation
        gravity_orientation = self.obs.get_foot('gravity')
        foot_positions = self.obs.get_foot('foot_position')
        leg_state = self.obs.get()

        qj_obs = leg_state["joint_positions"]
        dqj_obs = leg_state["joint_velocities"]
        ang_vel = leg_state["imu_omega"]

        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale  #joint offsets
        dqj_obs = dqj_obs * self.config.dof_vel_scale 

        ang_vel = ang_vel * self.config.ang_vel_scale
        num_actions = self.config.num_actions

        cmd = self.get_remote_analog() #self.lx, self.rx, self.ry, self.ly
        self.cmd_arr = cmd[[3, 0, 1]]  # Selecting elements in the order: ly -> X, lx -> Y, rx -> YAW

        self.obs[:12] = foot_positions 
        self.obs[12:15] = ang_vel
        self.obs[15:18] = gravity_orientation
        self.obs[18 : 18 + num_actions] = qj_obs
        self.obs[18 + num_actions : 18 + num_actions*2] = dqj_obs
        self.obs[18 + num_actions*2 : 18 + num_actions*3] = self._last_action
        self.obs[18 + num_actions*3: 18 + num_actions*3 + 3] = self.cmd_arr 

        onnx_input = {"state": self.obs.reshape(1, -1)}
        self.action = self._policy.run(None, onnx_input)[0][0]
        self._last_action = self.action.copy()
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd) 

        time.sleep(self.config.control_dt)

    #define what the robot do when emergency stopped
    def emergency(self):
        #rapidly move to default pos to avoid fall of robot
        self.move_to_pos(self.config.crounch_angles, total_time = 2, ask= False)
        self.zero_torque_state(ask=False)


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
    #ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_pos(config.default_angles)


    print(f"Press R1 to start RL-Controller, PRESS R2 for any emergenry")
    while controller.get_remote_digital[KeyMap.R1] != 1:
            time.sleep(controller.config.control_dt)


    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.get_remote_digital[KeyMap.R2] == 1:
                controller.emergency()
                break
        except KeyboardInterrupt:
            controller.emergency()
            break
   
    print("Safe_Exit")
