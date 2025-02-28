
from numba.pycc import CC
import numpy as np
import go2_deploy.utils.math_function as MF


cc = CC('Go2_estimation_CF')


"""# FREQUENCY SETTING
freq  = 500.  # run at 500 Hz
dt    = 1. / freq
dt2   = dt * dt
dt2_2 = dt2 / 2."""


@cc.export('run', '(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:],'
                                 'f8[:], f8[:], f8[:], f8[:],'
                                 'f8[:,:], f8[:], f8[:], f8[:], f8, f8[:])')
def run(v0_b, a0_b, p_fr_b, p_fl_b, p_rr_b, p_rl_b,  
                v_fr_b, v_fl_b, v_rr_b, v_rl_b,
                Rm, wm, am, foot_contacts, g, kv, dt):
    # POSITION AND VELOCITY ESTIMATE
    # predict
    v1_b = v0_b + a0_b * dt
    # update
    
    vc_b   = np.zeros(3)
    what = MF.hat(wm)
    Rm   = np.copy(Rm)
    total_contacts = 0

    if foot_contacts[0]:  #contact for FR_foot
        total_contacts += 1
        p_fr_b = np.copy(p_fr_b)
        vc_b -= Rm @ (what @ p_fr_b + v_fr_b)

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
    else:
        vc_b /= total_contacts #taking avrage of all velocities from contact legs
        for i in range(3):
            v1_b[i] = kv[i] * vc_b[i] + (1. - kv[i]) * v1_b[i]  #using compliment filer 
    
    a1  = Rm @ np.copy(am) - np.array([0., 0., g])  # body acceleration excluding gravity
    bv1 = Rm.T @ np.copy(v1_b)
    yaw = np.arctan2(Rm[1, 0], Rm[0, 0])

    return Rm, wm, v1_b, a1, bv1, yaw


if __name__ == '__main__':
    cc.compile()
