import go2_deploy.utils.publisher as pub
import numpy as np
import go2_deploy.utils.Go2_estimation_CF as EST
import go2_deploy.utils.Go2_kinematics_CF as KIN
import go2_deploy.utils.Go2_orientation_CF as ORI
from termcolor import colored
import go2_deploy.utils.math_function as MF
import os
from unitree_sdk2py.utils.thread import RecurrentThread

#the estimator will only estimate foot positions from joint positions
# Parameters
loop_freq     = 1500  # run at 500 Hz
loop_duration = 1. / loop_freq

# BRUCE Setup
state = pub.GO2STATE()


gravity_accel = 9.81

# Foot Contacts
foot_contacts       = np.zeros(4)  # 0/1 indicate in air/contact (for right/left toe/heel)
foot_contacts_count = np.zeros(4)  # indicate how long the foot is in contact

# Initial Guess

v_wb  = np.array([0., 0., 0.])         # body velocity         - in world frame
a_wb  = np.array([0., 0., 0.])         # body acceleration     - in world frame

R_wb  = np.eye(3)                      # body orientation      - in world frame
SEq_1, SEq_2, SEq_3, SEq_4 = 1.0, 0.0, 0.0, 0.0
v_bb  = R_wb.T @ v_wb                  # body velocity         - in  body frame
w_bb  = np.array([0., 0., 0.])         # body angular velocity - in  body frame
b_acc = np.array([0., 0., 0.])         # accelerometer bias    - in   IMU frame

R_wi = np.eye(3)                       # imu orientation in world frame
w_ii = np.zeros(3)                     # last gyroscope reading

Po = np.eye(15) * 1e-2                 # Kalman filter state covariance matrix

# Start Estimation
print("====== SM Diagnostics", loop_freq, "Hz... ======")

t0 = state.get_time()['time_stamp']
thread_run = False


def est_main():
    loop_start_time = state.get_time()['time_stamp']
    elapsed_time    = loop_start_time - t0

    # get info from shared memory
    leg_data = state.LEG_STATE.get()

    # get leg joint states
    q  = leg_data['joint_positions']
    dq = leg_data['joint_velocities']
    quat = leg_data['imu_ori']
    os.system("clear")
    print(f"q: {q}")
    print(f"dq: {dq}")
    print(f"quat: {quat}")

    print(f'foot_pos: {state.get_foot("foot_position")}')
    print(f'gravity: {state.get_foot("gravity")}')

    print(f'BUTTONS: {state.get_remote_data("Digital")}')

    #os.system("clear")
    #print(f'STICK: {state.get_remote_data("Analog")}')



def start_diag():
        lowCmdWriteThreadPtr = RecurrentThread(
            interval=loop_duration, target=est_main, name="writebasiccmd"
        )
        lowCmdWriteThreadPtr.Start()


if __name__ == "__main__":
    start_diag()

    while True:
         pass




       