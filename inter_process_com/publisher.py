from utils import memory_manager as shmx
import numpy as np
import posix_ipc
import time

class publish_cmd():

    def __init__(self) -> None:

        self.JOINT_POSITION_COMMAND = shmx.SHMEMSEG(robot_name='Go1', seg_name='CMD_MUJOCO', init=False)
        self.JOINT_POSITION_COMMAND.add_block(name='XYyaw', data=np.zeros(3))

        try:
            self.JOINT_POSITION_COMMAND.connect_segment()
            
        except posix_ipc.ExistentialError:
            self.JOINT_POSITION_COMMAND.initialize = True
            self.JOINT_POSITION_COMMAND.connect_segment()


    def set(self, position):

        data = {'XYyaw': np.array(position)}
        self.JOINT_POSITION_COMMAND.set(data)

    def get(self):

        return self.JOINT_POSITION_COMMAND.get()
    

class publish_joint_cmd():


    def __init__(self) -> None:

        self.JOINT_POSITION_COMMAND = shmx.SHMEMSEG(robot_name='Go2', seg_name='low_level', init=False)
        self.JOINT_POSITION_COMMAND.add_block(name='XYyaw', data=np.zeros(3))

        try:
            self.JOINT_POSITION_COMMAND.connect_segment()
            
        except posix_ipc.ExistentialError:
            self.JOINT_POSITION_COMMAND.initialize = True
            self.JOINT_POSITION_COMMAND.connect_segment()


    def set(self, position):

        data = {'joint_cmd': np.array(position)}
        self.JOINT_POSITION_COMMAND.set(data)

    def get(self):

        return self.JOINT_POSITION_COMMAND.get()

