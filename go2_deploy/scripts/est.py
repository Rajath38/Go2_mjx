import go2_deploy.utils.publisher as pub
import numpy as np
import go2_deploy.utils.Go2_estimation_CF as EST
import go2_deploy.utils.Go2_kinematics_CF as KIN
import go2_deploy.utils.Go2_orientation_CF as ORI
from termcolor import colored
import go2_deploy.utils.math_function as MF


#the estimator will only estimate foot positions from joint positions

def est_main():
    # BRUCE Setup
    state = pub.GO2STATE()
    
    # Parameters
    loop_freq     = 1500  # run at 500 Hz
    loop_duration = 1. / loop_freq
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
    print("====== The State Estimation Thread is running at", loop_freq, "Hz... ======")

    t0 = state.get_time()['time_stamp']
    thread_run = False

    while True:
        loop_start_time = state.get_time()['time_stamp']
        elapsed_time    = loop_start_time - t0

        # get info from shared memory
        leg_data = state.LEG_STATE.get()

        # get leg joint states
        q  = leg_data['joint_positions']
        dq = leg_data['joint_velocities']
        quat = leg_data['imu_ori']

        # compute leg forward kinematics
        pfr, vfr, Jfr, pfl, vfl, Jfl, prr, vrr, Jrr, prl, vrl, Jrl = KIN.legFK(q[0], q[1], q[2],  0,
                                                                              q[3], q[4], q[5],  0,
                                                                              q[6], q[7], q[8],  0,
                                                                              q[9], q[10],q[11], 0,
                                                                              dq[0], dq[1],  dq[2], 
                                                                              dq[3], dq[4],  dq[5], 
                                                                              dq[6], dq[7],  dq[8], 
                                                                              dq[9], dq[10], dq[11] )

        data = np.array([pfr, pfl, prr, prl])
        gravity = MF.get_gravity_orientation(quat)

        state.set_foot(data, "foot_positions")
        state.set_foot(gravity, "gravity")

        # check time to ensure that the state estimator stays at a consistent running loop.
        loop_end_time = loop_start_time + loop_duration
        present_time  = state.get_time()['time_stamp']
        if present_time > loop_end_time:
            delay_time = 1000 * (present_time - loop_end_time)
            if delay_time > 1.:
                print(colored('Delayed ' + str(delay_time)[0:5] + ' ms at Te = ' + str(elapsed_time)[0:5] + ' s', 'yellow'))
        else:
            while state.get_time()['time_stamp'] < loop_end_time:
                pass

       