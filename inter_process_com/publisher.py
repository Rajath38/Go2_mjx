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
    
class ThreadStatus():

    def __init__(self) -> None:

        self.status= shmx.SHMEMSEG(robot_name='Go2', seg_name='thread_status', init=False)
        self.status.add_block(name='est', data=np.zeros(1, dtype=bool))
        self.status.add_block(name='cmd', data=np.zeros(1, dtype=bool))
        self.status.add_block(name='obs', data=np.zeros(1, dtype=bool))

        try:
            self.status.connect_segment()
            
        except posix_ipc.ExistentialError:
            self.status.initialize = True
            self.status.connect_segment()

    def set(self, key, value:bool):
        data = {key: np.array(value)}
        self.status.set(data)

    def get(self):
        return self.status.get()


class GetSetObservations():

    """linvel, #3
        gyro, #3
        gravity, #3
        del_joint_angles, #12
        joint_velocities, #12
        self._last_action, #12
        self.PJ.get()['XYyaw'], #3"""

    def __init__(self) -> None:

        self.obs1 = shmx.SHMEMSEG(robot_name='Go2', seg_name='obs1', init=False)
        self.obs1.add_block(name='linvel', data=np.zeros(3))
        self.obs1.add_block(name='gravity', data=np.zeros(3))
        self.obs1.add_block(name='del_joint_angles', data=np.zeros(12))

        self.obs2 = shmx.SHMEMSEG(robot_name='Go2', seg_name='obs2', init=False)
        self.obs2.add_block(name='joint_vel', data=np.zeros(12))

        self.obs3= shmx.SHMEMSEG(robot_name='Go2', seg_name='obs4', init=False)
        self.obs3.add_block(name='cmd', data=np.zeros(3))
        self.obs3.add_block(name='acceleration', data=np.zeros(3))

        try:
            self.obs1.connect_segment()
            self.obs2.connect_segment()
            self.obs3.connect_segment()
    
        except posix_ipc.ExistentialError:
            self.obs1.initialize = True
            self.obs1.connect_segment()

            self.obs2.initialize = True
            self.obs2.connect_segment()

            self.obs3.initialize = True
            self.obs3.connect_segment()


    def set(self, obs: np.ndarray):

        data = { 'gravity': obs[3:6],
                'del_joint_angles': obs[6:18] }
        self.obs1.set(data)

        data = {'joint_vel' : obs[18:30]}
        self.obs2.set(data)

        data = {'cmd' : obs[42:45], 
                'acceleration': obs[45:48] }
        self.obs3.set(data)

    def set__estmator(self, obs: np.ndarray):
        data = { 'lin_vel': obs[:3]}
        self.obs1.set(data)

    def get(self):
        dict1 = self.obs1.get()
        dict2 = self.obs2.get()
        dict3 = self.obs3.get()
        merged_dict = dict1 | dict2 | dict3 
        return merged_dict



