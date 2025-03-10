from go2 import LEGGED_GYM_ROOT_DIR
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from go2_deploy.common.rotation_helper import get_gravity_orientation, transform_imu_data
from go2_deploy.common.remote_controller import RemoteController, KeyMap
from go2_deploy.config import Config
from go2_deploy.common.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.utils.thread import RecurrentThread
from inter_process_com.publisher import GetSetObservations 
from go2_deploy.utils.publisher import GO2STATE


#this subscribes to the go2 to get the low_states and add to the shared memory.


class Sense:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.get_set_obs = GetSetObservations()
        self.SM = GO2STATE()

        
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


    def run(self):

        # Get the current joint position and velocity
        for i in self.config.leg_joint2motor_idx:
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # record sensor data into SM
        
        qj_obs = self.qj.copy()  # joint positions
        dqj_obs = self.dqj.copy()  # joint velocities

        """
        gravity_orientation = get_gravity_orientation(quat) # joint velocities
        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel   #lin_vel is yet to be added soon form est_ data
        self.obs[3:6] = ang_vel
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions*2] = dqj_obs
        self.obs[9 + num_actions*2 : 9 + num_actions*3] = self.action
        self.obs[9 + num_actions*3: 9 + num_actions*3 + 3] = self.cmd 
        
        """

        self.SM.set_data(qj_obs, "joint_positions")
        self.SM.set_data(dqj_obs, "joint_velocities")
        self.SM.set_data(quat, "imu_ori")
        self.SM.set_data(ang_vel, "imu_omega")



