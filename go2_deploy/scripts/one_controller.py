from go2 import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.utils.crc import CRC

from go2_deploy.common.command_helper import create_zero_cmd, init_cmd_go, create_damping_cmd
from go2_deploy.common.remote_controller import KeyMap
from go2_deploy.config import Config
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from go2_deploy.common.motion_switcher_client import MotionSwitcherClient
from go2_deploy.common.remote_controller import RemoteController
import go2_deploy.utils.Go2_kinematics_CF as KIN
import go2_deploy.utils.math_function as MF
from unitree_sdk2py.utils.thread import RecurrentThread
import sys

from go2_deploy.utils.publisher import GO2STATE #only for diagnostics 

import onnxruntime as rt
import argparse

#dummy controller to test program flow and emergency stops


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.loop_freq = 2500 #frequency of obs loop in Hz, anything less than this , is lesser than the actual subscription callback, ehich results in non Real time update of states 
        #self.qj = config.default_angles #np.zeros(12) is not recommended as it causes problems 
        #Important about numpy is above assignment
        #since we are initializing this self.qj = config.default_angles, they are just the same array addressed by different names
        # so later in the obs_thread, as we are changing self.qj, we cal also observe that config.default_angles also change
        # crrating lot of issues, so we have to make a conpy assignment here.
        self.qj = np.copy(config.default_angles) 
        self.dqj = np.zeros(12)
   
        self.obs = np.zeros(config.num_obs, dtype=np.float32)

        # go2 uses the go msg type
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        self.SM = GO2STATE()

        self.remote_controller = RemoteController()

        ChannelFactoryInitialize(0, "enp2s0")

        #initialize a Publisher
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)  # Increase buffer size
        

        self.action = np.zeros(self.config.num_actions)
        self.last_action = np.zeros(self.config.num_actions)
        self.last_last_action = np.zeros(self.config.num_actions)
        #init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        policy_path=config.policy_path

        self.action_scale=0.4
        self._last_action = np.zeros_like(self.action, dtype=np.float32)
        self.motor_targets = np.zeros(12)
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self.motor_targets_min_limit = np.array([
                                                -0.25,  0.52, -2.12, -0.45,  0.51, -2.15,  
                                                -0.22,  0.74, -2.14, -0.35,  0.69, -2.16
                                            ])

        self.motor_targets_max_limit = np.array([
                                                0.42,  1.24, -1.40,  0.25,  1.26, -1.40,  
                                                0.35,  1.25, -1.40,  0.25,  1.22, -1.40
                                            ])
        

  
        # go2 uses the go msg type
        self.low_state = unitree_go_msg_dds__LowState_()

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

        init_cmd_go(self.low_cmd)

        self.start_obs()
        #wait for 5 sec for the obs thread to initialize
        time.sleep(10)


    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
        
    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")


    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = self.crc.Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)


    def zero_torque_state(self, ask = True):
       
        if ask == True:
            print("For zero torque state.")
            print("Enter 'A'...to proceed")
            while self.remote_controller.button[KeyMap.A] != 1:
                time.sleep(self.config.control_dt)

        print("Zero torque")
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
    
    def move_to_pos(self, default_pos, total_time = 2, ask = True):
        # move time 2s
        error_status = False
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps = self.config.kps 
        kds = self.config.kds 
        
        dof_size = len(dof_idx)

        if ask == True:
            print("For Default Pose.")
            print("Enter 'B'...to proceed")
            while self.remote_controller.button[KeyMap.B] != 1:
                time.sleep(self.config.control_dt)
                print(f"dof_pos: {self.qj[2]}")
                print(f"low_state.motor_state[2].q: {self.low_state.motor_state[2].q}")

        # record the current pos, here anagin since self.qj is updated by obs thread instantaniousle
        # we need to take a copy of this as init_dof_pos and then move to the default pose.
        init_dof_pos = np.copy(self.qj)
        print(f"Initial: {init_dof_pos}")
        print(f"Final: {default_pos}")
            
        # move to default pos
        print(f"Moving to requested robot pose")

        # move to default pos
        for i in range(num_step):
            if self.remote_controller.button[KeyMap.R2] == 1:
                print(f"EMERGENCY PRESSED")
                self.emergency()
                return True
        
            print(f"Moving step {i}")
            alpha = i / num_step

            for j in range(dof_size): #comment to only move calf_FL
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                inst_pos = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                print(f"Inst_pos:{inst_pos}, alpha:{alpha}")
                self.low_cmd.motor_cmd[motor_idx].q = inst_pos
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        
        if ask == True:
            error_status = self.hold_default_pos_state()

        return error_status


    def hold_default_pos_state(self):

        print(f"Press R1 to start RL-Controller, PRESS R2 for any emergenry")
        while self.remote_controller.button[KeyMap.R1] != 1: 
            if self.remote_controller.button[KeyMap.R2] == 1:
                print(f"EMERGENCY PRESSED")
                self.emergency()
                return True
            
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

        return False
            


    def obs_thread(self):
        # compute leg forward kinematics
        pfr, _, _, pfl, _, _, prr, _, _, prl, _, _ = KIN.legFK( self.low_state.motor_state[0].q,  self.low_state.motor_state[1].q,   self.low_state.motor_state[2].q,  0,
                                                                                self.low_state.motor_state[3].q,  self.low_state.motor_state[4].q,   self.low_state.motor_state[5].q,  0,
                                                                                self.low_state.motor_state[6].q,  self.low_state.motor_state[7].q,   self.low_state.motor_state[8].q,  0,
                                                                                self.low_state.motor_state[9].q,  self.low_state.motor_state[10].q,  self.low_state.motor_state[11].q, 0,
                                                                                self.low_state.motor_state[0].dq, self.low_state.motor_state[1].dq,  self.low_state.motor_state[2].dq, 
                                                                                self.low_state.motor_state[3].dq, self.low_state.motor_state[4].dq,  self.low_state.motor_state[5].dq, 
                                                                                self.low_state.motor_state[6].dq, self.low_state.motor_state[7].dq,  self.low_state.motor_state[8].dq, 
                                                                                self.low_state.motor_state[9].dq, self.low_state.motor_state[10].dq, self.low_state.motor_state[11].dq)
        self.foot_pos = np.array([pfr, pfl, prr, prl])
        self.foot_positions = np.ravel(self.foot_pos)
        self.ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        quat = np.array(self.low_state.imu_state.quaternion)
        quat_ = quat[[1,2,3,0]] #from w, x, y, z to  x, y, z, w 
        imu_xmat = MF.quat2rotm(quat_)
        self.gravity_orientation = imu_xmat.T @ np.array([0, 0, -1])

        # Get the current joint position and velocity
        for i in self.config.leg_joint2motor_idx:
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        self.cmd_arr = np.array([self.remote_controller.ry, self.remote_controller.rx, self.remote_controller.lx])  # Selecting elements in the order: ry -> X, rx -> Y, lx -> YAW

        self.SM.set_data(self.qj, "joint_positions")
        self.SM.set_data(self.dqj, "joint_velocities")
        self.SM.set_data(quat, "imu_ori")
        self.SM.set_data(self.ang_vel, "imu_omega")

        self.SM.set_foot(self.foot_positions, "foot_position")
        self.SM.set_foot(self.gravity_orientation, "gravity")

    
    
    
    def start_obs(self):
        print(f"====== The Sense Thread is running at {self.loop_freq} Hz... ======")
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.loop_freq, target=self.obs_thread, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()


    def run(self):

        """ noisy_feet_pos, # 12 # if we remove this legs dont touch properly with floor
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["command"],  # 3"""
        num_actions = self.config.num_actions

        if self.remote_controller.button[KeyMap.R2] == 1:
            print(f"EMERGENCY PRESSED")
            self.emergency()
            return True
        qj_offsets = (self.qj - self.config.default_angles) * self.config.dof_pos_scale  #joint offsets

        self.obs[:12] = self.foot_positions 
        self.obs[12:15] = self.ang_vel*self.config.ang_vel_scale
        self.obs[15:18] = self.gravity_orientation
        self.obs[18 : 18 + num_actions] = qj_offsets
        self.obs[18 + num_actions : 18 + num_actions*2] = self.dqj*self.config.dof_vel_scale
        self.obs[18 + num_actions*2 : 18 + num_actions*3] = self._last_action
        self.obs[18 + num_actions*3: 18 + num_actions*3 + 3] = self.cmd_arr 

        print(f"cmd:{self.cmd_arr}")

        onnx_input = {"state": self.obs.reshape(1, -1)}
        self.action = self._policy.run(None, onnx_input)[0][0]
        self._last_action = self.action.copy()
        # transform action to target_dof_pos
        self.motor_targets = self.config.default_angles + self.action * self.config.action_scale
        clipped_array, fault = self.clipped_and_fault(self.motor_targets)
        #put safe limits on the targett joint positions

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = clipped_array[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        print(f"target_position:{clipped_array}") 
        print(f"fault: {fault}")

        """if np.any(fault) == True:
            print(f"Exit due to dof limit exceed")
            return True"""

        time.sleep(self.config.control_dt)
        
        return False

    #define what the robot do when emergency stopped
    def emergency(self):
        #unitree implimentation had zero damping not sure if its better ....
        #create_damping_cmd(controller.low_cmd)
        #self.send_cmd(controller.low_cmd)
        # not sure if damping is better than moving to corunch position, we can check by running experiments.... still pending, hardware not available ....
          #rapidly move to crounch pos to avoid fall of robot
        #self.move_to_pos(self.config.crounch_angles, total_time = 0.5, ask= False)
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        #self.zero_torque_state(ask=False)

    def clipped_and_fault(self, motor_torque):

        clipped  = np.clip(motor_torque, self.motor_targets_min_limit, self.motor_targets_max_limit)
        fault = (clipped == self.motor_targets_max_limit)|(clipped == self.motor_targets_min_limit)
        return clipped, fault
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{args.config}"
   
    config = Config(config_path)

    controller = Controller(config)

    controller.zero_torque_state()

    print(f"default angles: {config.default_angles}")

    # Move to the default position
    fault = controller.move_to_pos(config.default_angles)

    if fault:
        print(f"Fault in startup")
        sys.exit(1)


    print("====== The CONTROLLER is running at", (1/config.control_dt), "Hz... ======")

    while True:
        try:
            fault = controller.run()
            if fault:
                break
            
        except KeyboardInterrupt:
            controller.emergency()
            break
   
    print("Safe_Exit")
