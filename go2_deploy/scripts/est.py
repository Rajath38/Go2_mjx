import go2_deploy.utils.publisher as pub
import numpy as np
import go2_deploy.utils.Go2_estimation_CF as EST
import go2_deploy.utils.Go2_kinematics_CF as KIN
import go2_deploy.utils.Go2_orientation_CF as ORI
from termcolor import colored
import go2_deploy.utils.math_function as MF
from unitree_sdk2py.utils.thread import RecurrentThread


HARDWARE = False

# System constants
deltat = 1/1500  # Sampling period in seconds (1 ms)
gyro_meas_error = np.pi * (0.0 / 180.0)  # Gyroscope measurement error in rad/s (5 deg/s)
beta = np.sqrt(3.0 / 4.0) * gyro_meas_error  # Compute beta

def main_loop():
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

    # Shared Memory Data
    estimation_data = {'body_rot_matrix': np.zeros((3, 3))}

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

        # compute leg forward kinematics
        pfr, vfr, Jfr, pfl, vfl, Jfl,prr, vrr, Jrr, prl, vrl, Jrl = KIN.legFK(q[0], q[1], q[2],  0,
                                                                              q[3], q[4], q[5],  0,
                                                                              q[6], q[7], q[8],  0,
                                                                              q[9], q[10],q[11], 0,
                                                                              dq[0], dq[1],  dq[2], 
                                                                              dq[3], dq[4],  dq[5], 
                                                                              dq[6], dq[7],  dq[8], 
                                                                              dq[9], dq[10], dq[11] )

        # state estimation
      
        foot_contacts = state.FOOT_STATE.get()['foot_contact']
        for idx in range(4):
            foot_contacts_count[idx] = foot_contacts_count[idx] + 1 if foot_contacts[idx] else 0

        #print(f"IMU: {leg_data['imu_omega'], leg_data['imu_acc']}")
        
        # robot mode - balance 0
    #              walking 1

        """if Bruce.mode == 0:
            kR = 0.002
        elif Bruce.mode == 1:
            kR = 0.001"""
        
        kR = 0.005

        sim_dt = 0.005#0.005 #0.00025 #simulation advances by 0.25ms
        control_dt = 0.02#0.025 #0.005#0.025  #but the update in made every 25ms

        """R_wi, w_ii = ORI.run(R_wi, w_ii,
                             leg_data['imu_omega'], leg_data['imu_acc'], gravity_accel, kR, sim_dt)"""
        
        R_wi, w_ii = ori_run(R_wi, w_ii,
                             leg_data['imu_omega'], leg_data['imu_acc'], gravity_accel, kR, sim_dt)
        
        SEq_1, SEq_2, SEq_3, SEq_4 = filter_update(leg_data['imu_omega'][0], leg_data['imu_omega'][1], leg_data['imu_omega'][2], 
                              leg_data['imu_acc'][0], leg_data['imu_acc'][1], leg_data['imu_acc'][2], 
                              SEq_1, SEq_2, SEq_3, SEq_4)

        
        kp, kv = np.array([0.1, 0.1, 0.1]), np.array([0.8, 0.8, 0.8])


        """R_wb, w_bb, v_wb, a_wb, \
        v_bb, yaw_angle = EST.run(v_wb, a_wb, pfr, pfl, prr, prl,  
                                               np.expand_dims(vfr, axis=1), vfl, vrr, vrl, 
                                  R_wi, w_ii, leg_data['imu_acc'], foot_contacts, gravity_accel,
                                         kv, sim_dt)"""

        #print(f"BEFORE v_wb: {v_wb}")
        R_wb, w_bb, v_wb, a_wb, \
        v_bb, yaw_angle = run(v_wb, a_wb, pfr, pfl, prr, prl,  
                        np.expand_dims(vfr, axis=1),  np.expand_dims(vfl, axis=1),  np.expand_dims(vrr, axis=1),  np.expand_dims(vrl, axis=1),
                                  R_wi, w_ii, leg_data['imu_acc'], foot_contacts, gravity_accel,
                                         kv, sim_dt)


        
       

        #print(f"velocity = {v_wb}")

        Q = rotation_matrix_to_quaternion(R_wi)
        print(f"Base_orientation_old: {Q}")
        #print(f"Base_orientation_updated: {SEq_1.item(), SEq_2.item(), SEq_3.item(), SEq_4.item()}")


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


if __name__ == '__main__':
    main_loop()
       