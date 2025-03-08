import go2_deploy.utils.publisher as pub
import numpy as np
import go2_deploy.utils.Go2_estimation_CF as EST
import go2_deploy.utils.Go2_kinematics_CF as KIN
import go2_deploy.utils.Go2_orientation_CF as ORI
from termcolor import colored
import go2_deploy.utils.math_function as MF


HARDWARE = False

# System constants
deltat = 1/1500  # Sampling period in seconds (1 ms)
gyro_meas_error = np.pi * (0.0 / 180.0)  # Gyroscope measurement error in rad/s (5 deg/s)
beta = np.sqrt(3.0 / 4.0) * gyro_meas_error  # Compute beta

def run(v0_b, a0_b, p_fr_b, p_fl_b, p_rr_b, p_rl_b,  
                v_fr_b, v_fl_b, v_rr_b, v_rl_b,
                Rm, wm, am, foot_contacts, g, kv, dt):
    # POSITION AND VELOCITY ESTIMATE
    # predict
    v1_b = v0_b + a0_b * dt
    #print(f"v1b:{v1_b}")
    #print(f"v0_b:{v0_b}")
    #print(f"a0_b:{a0_b}")
    # update
    
    vc_b   = np.zeros((3,1))
    what = MF.hat(wm)
    Rm   = np.copy(Rm)
    #print(f"Rm:{Rm}")
    total_contacts = 0

    if foot_contacts[0]:  #contact for FR_foot
        total_contacts += 1
        p_fr_b = np.copy(p_fr_b)
        #print(f"prb:{p_fr_b, v_fr_b}")
        #print((what @ p_fr_b))
        vc_b -= Rm @ (what @ p_fr_b + v_fr_b)
        #print(f"vc_b:{vc_b}")

    if foot_contacts[1]:  #contact for FL_foot
        total_contacts += 1
        p_fl_b = np.copy(p_fl_b)
        vc_b -= Rm @ (what @ p_fl_b + v_fl_b)

    if foot_contacts[2]:  #contact for RR_foot
        total_contacts += 1
        p_rr_b = np.copy(p_rr_b)
        vc_b -= Rm @ (what @ p_rr_b + v_rr_b)

    if foot_contacts[3]: #contact for RL_foot
        total_contacts += 1
        p_rl_b = np.copy(p_rl_b)
        vc_b -= Rm @ (what @ p_rl_b + v_rl_b)


    if total_contacts == 0:  # do nothing if lose all contacts to prevent divergence
        v1_b = v0_b
        #print(f"ALL CONTACTS LOST, v1b: {v1_b}")
    else:
        vc_b /= total_contacts #taking avrage of all velocities from contact legs
        for i in range(3):
            v1_b[i] = kv[i] * vc_b[i] + (1 - kv[i]) * v1_b[i]  #using compliment filer 
            #print(v1_b[i])
        #print(f"SOME CONTACTS, v1b: {v1_b}")
    
    a1  = Rm @ np.copy(am) - np.array([0., 0., g])  # body acceleration excluding gravity
    bv1 = Rm.T @ np.copy(v1_b)
    yaw = np.arctan2(Rm[1, 0], Rm[0, 0])

    return Rm, wm, v1_b, a1, bv1, yaw

def ori_run(R0, w0,
        omg, acc, g, kR, dt):
    # predict
    R1 = np.copy(R0) @ MF.hatexp(w0 * dt)
    w1 = omg

    # update
    gm   = R1 @ np.copy(acc)
    gmu  = gm / MF.norm(gm)
    dphi = np.arccos(gmu[2] * np.sign(g))
    nv   = np.zeros(3) if np.abs(dphi) < 1e-10 else MF.hat(gmu) @ np.array([0., 0., np.sign(g)]) / np.sin(dphi)  # rotation axis
    R1   = MF.hatexp(kR * dphi * nv) @ R1

    return R1, w1

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].
    
    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix

    Returns:
        numpy.ndarray: Quaternion [qx, qy, qz, qw]
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def filter_update(w_x, w_y, w_z, a_x, a_y, a_z, SEq_1, SEq_2, SEq_3, SEq_4):

    # Normalize accelerometer measurement
    norm = np.sqrt(a_x * a_x + a_y * a_y + a_z * a_z)
    a_x /= norm
    a_y /= norm
    a_z /= norm

    # Auxiliary variables
    halfSEq_1, halfSEq_2, halfSEq_3, halfSEq_4 = 0.5 * SEq_1, 0.5 * SEq_2, 0.5 * SEq_3, 0.5 * SEq_4
    twoSEq_1, twoSEq_2, twoSEq_3 = 2.0 * SEq_1, 2.0 * SEq_2, 2.0 * SEq_3

    # Compute objective function and Jacobian
    f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x
    f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y
    f_3 = 1.0 - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z

    J_11or24, J_12or23, J_13or22, J_14or21 = twoSEq_3, 2.0 * SEq_4, twoSEq_1, twoSEq_2
    J_32, J_33 = 2.0 * J_14or21, 2.0 * J_11or24

    # Compute the gradient (matrix multiplication)
    SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1
    SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3
    SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1
    SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2

    # Normalize the gradient
    norm = np.sqrt(SEqHatDot_1**2 + SEqHatDot_2**2 + SEqHatDot_3**2 + SEqHatDot_4**2)
    SEqHatDot_1 /= norm
    SEqHatDot_2 /= norm
    SEqHatDot_3 /= norm
    SEqHatDot_4 /= norm

    # Compute quaternion derivative measured by gyroscopes
    SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z
    SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y
    SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x
    SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x

    # Compute then integrate the estimated quaternion derivative
    SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * deltat
    SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * deltat
    SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * deltat
    SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * deltat

    # Normalize quaternion
    norm = np.sqrt(SEq_1**2 + SEq_2**2 + SEq_3**2 + SEq_4**2)
    SEq_1 /= norm
    SEq_2 /= norm
    SEq_3 /= norm
    SEq_4 /= norm

    return SEq_1, SEq_2, SEq_3, SEq_4

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
       