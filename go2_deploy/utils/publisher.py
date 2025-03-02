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
        self.LEG_STATE.add_block(name='imu_acc', data=np.zeros(3))
        self.LEG_STATE.add_block(name='imu_omega', data=np.zeros(3))

        self.FOOT_STATE = shmx.SHMEMSEG(robot_name='GO3', seg_name='FOOT_STATE', init=False)
        self.FOOT_STATE.add_block(name='foot_contact', data=np.zeros(4))

        # Simulator State
        self.SIMULATOR_STATE = shmx.SHMEMSEG(robot_name='GO4', seg_name='SIMULATOR_STATE', init=False)
        self.SIMULATOR_STATE.add_block(name='time_stamp', data=np.zeros(1))

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

    def set_data(self, data, data_type):

        data =  {data_type: np.array(data)}
        self.LEG_STATE.set(data)

    def get_data(self, data_type):
        data = self.LEG_STATE.get()
        return data[data_type]
    
    def get(self):
        return self.LEG_STATE.get()
    
    def set_foot(self, data):
        data_ = {'foot_contact': data}
        self.FOOT_STATE.set(data_)

    def get_foot(self):
        return self.FOOT_STATE.get()
    
    def set_time(self, data):
        data_ = {'time_stamp': data}
        self.SIMULATOR_STATE.set(data_)

    def get_time(self):
        return self.SIMULATOR_STATE.get()
    


    


