import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from go2_deploy.common.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import math

#MACROS

LegID = {
    "FR_0": 0,  # Front right hip
    "FR_1": 1,  # Front right thigh
    "FR_2": 2,  # Front right calf
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0


class Custom:
    def __init__(self):

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        
        self.qInit = [0.0] * 3
        self.qDes = [0.0] * 3
        self.sin_mid_q = [0.0, 1.2, -2.0]
        self.Kp = [0.0] * 3
        self.Kd = [0.0] * 3
        self.time_consume = 0
        self.rate_count = 0
        self.sin_count = 0
        self.motiontime = 0
        self.dt = 0.002
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None 
        self.crc = CRC()
        
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.lowCmdWriteThread = None

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            print(f"Trying to deactivate the motion control-related service..")
            self.sc.StandDown()
            code, _ = self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

    def joint_linear_interpolation(self, init_pos, target_pos, rate):
        rate = max(0.0, min(rate, 1.0))
        return init_pos * (1 - rate) + target_pos * rate
    
    def LowCmdWrite(self):

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
            freq_rad = freq_Hz * 2 * math.pi
            sin_joint1 = 0.6 * math.sin(t * freq_rad)
            sin_joint2 = -0.9 * math.sin(t * freq_rad)
            
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

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    input("Initialization complete. Press Enter to start the motion...")
    
    custom.Start()

    while True:         
        time.sleep(25)