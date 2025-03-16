from go2 import LEGGED_GYM_ROOT_DIR
import numpy as np
import time

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from go2_deploy.common.remote_controller import RemoteController
from go2_deploy.config import Config
from go2_deploy.common.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from go2_deploy.utils.publisher import GO2STATE
from termcolor import colored
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.utils.thread import RecurrentThread

#this subscribes to the go2 to get the low_states and add to the shared memory.


class Sense:
    def __init__(self, config: Config) -> None:

        self.loop_freq = 1000 # Hz
        self.config = config
        self.remote_controller = RemoteController()
        self.SM = GO2STATE()

        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        ChannelFactoryInitialize(0, "enp2s0")
        self.last_callback_time = time.time()  # Initialize timestamp

        #initialize  a subscriber
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        print("Initializing subscriber...")
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)  # Increase buffer size
        print("Subscriber initialized.")
        

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

        #init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        self.start()

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        
    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def sense_SM(self):

        self.remote_controller.set(self.low_state.wireless_remote)

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

        """ -----------------------FOR RFERENCE -------------------------------
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


    def start(self):
        print(f"====== The Sense Thread is running at {self.loop_freq} Hz... ======")
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=1/self.loop_freq, target=self.sense_SM, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()




def sense_main(config):

#if __name__ == "__main__":

    print("sense_main_1")
    

    # Load config
    #config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{config}"
    #config_path = "go2_deploy/configs/go2.yaml"
   
    #config = Config(config_path)

    #ChannelFactoryInitialize(0, "enp2s0")
    #lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)

    # sensor Setup
    Sense(config)

    while True:
        pass

