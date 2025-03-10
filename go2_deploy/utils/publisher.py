import utils.memory_manager as shmx
import numpy as np
import posix_ipc
import time
 
class GO2STATE():

    def __init__(self) -> None:

        # Leg State
        self.LEG_STATE = shmx.SHMEMSEG(robot_name='GO2', seg_name='STATE', init=False)
        self.LEG_STATE.add_block(name='joint_positions',  data=np.zeros(12))
        self.LEG_STATE.add_block(name='joint_velocities', data=np.zeros(12))
        self.LEG_STATE.add_block(name='imu_ori', data=np.zeros(4))
        self.LEG_STATE.add_block(name='imu_omega', data=np.zeros(3))

        self.FOOT_STATE = shmx.SHMEMSEG(robot_name='GO2', seg_name='FOOT_STATE', init=False)
        self.FOOT_STATE.add_block(name='foot_contact', data=np.zeros(4))
        self.FOOT_STATE.add_block(name='foot_position', data=np.zeros(12))
        self.FOOT_STATE.add_block(name='gravity', data=np.zeros(3))

        # Simulator State
        self.SIMULATOR_STATE = shmx.SHMEMSEG(robot_name='GO2', seg_name='SIMULATOR_STATE', init=False)
        self.SIMULATOR_STATE.add_block(name='time_stamp', data=np.zeros(1))

        # Remote Controller State
        self.REMOTE_STATE = shmx.SHMEMSEG(robot_name='GO2', seg_name='REMOTE_STATE', init=False)
        self.REMOTE_STATE.add_block(name='Digital', data=np.zeros(16))
        self.REMOTE_STATE.add_block(name='Analog', data=np.zeros(4))

        try:
            self.LEG_STATE.connect_segment()
            self.FOOT_STATE.connect_segment()
            self.SIMULATOR_STATE.connect_segment()
            
        except posix_ipc.ExistentialError:
            self.LEG_STATE.initialize = True
            self.LEG_STATE.connect_segment()

            self.FOOT_STATE.initialize = True
            self.FOOT_STATE.connect_segment()

            self.SIMULATOR_STATE.initialize = True
            self.SIMULATOR_STATE.connect_segment()
    
    def set_remote_data(self, data, data_type):
        data =  {data_type: np.array(data)}
        self.REMOTE_STATE.set(data)

    def get_remote_data(self, data_type):
        data = self.REMOTE_STATE.get()
        return data[data_type]
    
    def set_data(self, data, data_type):
        data =  {data_type: np.array(data)}
        self.LEG_STATE.set(data)

    def get_data(self, data_type):
        data = self.LEG_STATE.get()
        return data[data_type]
    
    def get(self):
        return self.LEG_STATE.get()
    
    def set_foot(self, data, data_type):
        data_ = {data_type: data}
        self.FOOT_STATE.set(data_)

    def get_foot(self, data_type):
        data = self.FOOT_STATE.get()
        return data[data_type]
    
    def set_time(self, data):
        data_ = {'time_stamp': data}
        self.SIMULATOR_STATE.set(data_)

    def get_time(self):
        return self.SIMULATOR_STATE.get()
    


    


