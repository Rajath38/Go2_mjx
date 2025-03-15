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
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber


#this subscribes to the go2 to get the low_states and add to the shared memory.


class Sense:
    def __init__(self, config: Config) -> None:
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

        #initialize  a subscriber
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        print("Initializing subscriber...")
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
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
        print(f"OKOKOKOKOK")
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
        print("Received a message in LowStateGoHandler!")  # Debugging
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    """def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.run, name="writebasiccmd"
        )
        print("Enter START to run main controller")
        while self.remote_controller.button[KeyMap.Y] != 1:
            pass 

        self.lowCmdWriteThreadPtr.Start()"""


def sense_loop(Sense: Sense):

    loop_freq = 1000  # run at 1000 Hz
    loop_duration = 1. / loop_freq


    print("====== The SENSOR COMMUNITCATION is running at", loop_freq, "Hz... ======")

    t0 = time.time()
    while True:
        loop_start_time = time.time()
        elapsed_time    = loop_start_time - t0

        # Get the current joint position and velocity
        for i in Sense.config.leg_joint2motor_idx:
            Sense.qj[i] = Sense.low_state.motor_state[i].q
            Sense.dqj[i] = Sense.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = Sense.low_state.imu_state.quaternion
        ang_vel = np.array([Sense.low_state.imu_state.gyroscope], dtype=np.float32)

        # record sensor data into SM
        
        qj_obs = Sense.qj.copy()  # joint positions
        dqj_obs = Sense.dqj.copy()  # joint velocities

        """
        gravity_orientation = get_gravity_orientation(quat) # joint velocities
        num_actions = Sense.config.num_actions
        Sense.obs[:3] = ang_vel   #lin_vel is yet to be added soon form est_ data
        Sense.obs[3:6] = ang_vel
        Sense.obs[6:9] = gravity_orientation
        Sense.obs[9 : 9 + num_actions] = qj_obs
        Sense.obs[9 + num_actions : 9 + num_actions*2] = dqj_obs
        Sense.obs[9 + num_actions*2 : 9 + num_actions*3] = Sense.action
        Sense.obs[9 + num_actions*3: 9 + num_actions*3 + 3] = Sense.cmd 
        
        """

        Sense.SM.set_data(qj_obs, "joint_positions")
        Sense.SM.set_data(dqj_obs, "joint_velocities")
        Sense.SM.set_data(quat, "imu_ori")
        Sense.SM.set_data(ang_vel, "imu_omega")


         # check time to ensure that the motor controller stays at a consistent control loop.
        loop_end_time = loop_start_time + loop_duration
        present_time  = time.time()
        if present_time > loop_end_time:
            delay_time = 1000. * (present_time - loop_end_time)
            if delay_time > 1.:
                print(colored('Delayed ' + str(delay_time)[0:5] + ' ms at Te = ' + str(elapsed_time)[0:5] + ' s', 'yellow'))
        else:
            while time.time() < loop_end_time:
                pass



def sense_main(config):

#if __name__ == "__main__":

    print("sense_main_1")
    

    """ # Load config
    #config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{config}"
    config_path = "go2_deploy/configs/go2.yaml"
   
    config = Config(config_path)

    ChannelFactoryInitialize(0, "enp2s0")
    lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)"""

    # sensor Setup
    sensor_read = Sense(config)

    print("sense_main_2")

    sense_loop(sensor_read)

