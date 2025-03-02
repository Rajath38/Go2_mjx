#!usr/bin/env python
from numba.pycc import CC
import numpy as np
from numpy import cos, sin


cc = CC('Go2_kinematics_CF')


@cc.export('legFK', '(f8, f8, f8, f8,'
                    ' f8, f8, f8, f8,'
                    ' f8, f8, f8, f8,'
                    ' f8, f8, f8, f8,'
                    ' f8, f8, f8,'
                    ' f8, f8, f8,'
                    ' f8, f8, f8,'
                    ' f8, f8, f8 )')

def legFK(frt1, frt2, frt3, frr,
          flt1, flt2, flt3, flr,
          rrt1, rrt2, rrt3, rrr,
          rlt1, rlt2, rlt3, rlr,
          frd1, frd2, frd3, 
          fld1, fld2, fld3,
          rrd1, rrd2, rrd3,
          rld1, rld2, rld3):
    
    #-----FRONT RIGHT LEG-------------------------------------------------------------------------------------
    
    theta1 = frt1
    theta2 = frt2
    theta3 = frt3

    fr_root_radius = frr
    dqfr = np.array([frd1, frd2, frd3])

    pfr = np.array([[-1.0*fr_root_radius*sin(theta2 + theta3) - 0.213*sin(theta2) - 0.213*sin(theta2 + theta3) + 0.21897],
       [1.0*(1.0*fr_root_radius*cos(theta2 + theta3) + 0.213*cos(theta2) + 0.213*cos(theta2 + theta3))*sin(theta1) - 0.0955*cos(theta1) - 0.0465],
       [-1.0*(1.0*fr_root_radius*cos(theta2 + theta3) + 0.213*cos(theta2) + 0.213*cos(theta2 + theta3))*cos(theta1) - 0.0955*sin(theta1) - 0.04232]])
    
    Jfr = np.array([[0,
        -1.0*fr_root_radius*cos(theta2 + theta3) - 0.213*cos(theta2) - 0.213*cos(theta2 + theta3),
        -1.0*fr_root_radius*cos(theta2 + theta3) - 0.213*cos(theta2 + theta3)],
       [1.0*(1.0*fr_root_radius*cos(theta2 + theta3) + 0.213*cos(theta2) + 0.213*cos(theta2 + theta3))*cos(theta1) + 0.0955*sin(theta1),
        1.0*(-1.0*fr_root_radius*sin(theta2 + theta3) - 0.213*sin(theta2) - 0.213*sin(theta2 + theta3))*sin(theta1),
        1.0*(-1.0*fr_root_radius*sin(theta2 + theta3) - 0.213*sin(theta2 + theta3))*sin(theta1)],
       [1.0*(1.0*fr_root_radius*cos(theta2 + theta3) + 0.213*cos(theta2) + 0.213*cos(theta2 + theta3))*sin(theta1) - 0.0955*cos(theta1),
        -1.0*(-1.0*fr_root_radius*sin(theta2 + theta3) - 0.213*sin(theta2) - 0.213*sin(theta2 + theta3))*cos(theta1),
        -1.0*(-1.0*fr_root_radius*sin(theta2 + theta3) - 0.213*sin(theta2 + theta3))*cos(theta1)]])
    
    #feet_linear velocities
    vfr = Jfr @ dqfr


    #-----FRONT LEFT LEG-------------------------------------------------------------------------------------
    
    theta4 = flt1
    theta5 = flt2
    theta6 = flt3
    fl_root_radius = flr
    dqfl = np.array([fld1, fld2, fld3])

    # position
    pfl = np.array([[-1.0*fl_root_radius*sin(theta5 + theta6) - 0.213*sin(theta5) - 0.213*sin(theta5 + theta6) + 0.21897],
       [1.0*(1.0*fl_root_radius*cos(theta5 + theta6) + 0.213*cos(theta5) + 0.213*cos(theta5 + theta6))*sin(theta4) + 0.0955*cos(theta4) + 0.0465],
       [-1.0*(1.0*fl_root_radius*cos(theta5 + theta6) + 0.213*cos(theta5) + 0.213*cos(theta5 + theta6))*cos(theta4) + 0.0955*sin(theta4) - 0.04232]])

    #Jacobian
    Jfl = np.array([[0,
        -1.0*fl_root_radius*cos(theta5 + theta6) - 0.213*cos(theta5) - 0.213*cos(theta5 + theta6),
        -1.0*fl_root_radius*cos(theta5 + theta6) - 0.213*cos(theta5 + theta6)],
       [1.0*(1.0*fl_root_radius*cos(theta5 + theta6) + 0.213*cos(theta5) + 0.213*cos(theta5 + theta6))*cos(theta4) - 0.0955*sin(theta4),
        1.0*(-1.0*fl_root_radius*sin(theta5 + theta6) - 0.213*sin(theta5) - 0.213*sin(theta5 + theta6))*sin(theta4),
        1.0*(-1.0*fl_root_radius*sin(theta5 + theta6) - 0.213*sin(theta5 + theta6))*sin(theta4)],
       [1.0*(1.0*fl_root_radius*cos(theta5 + theta6) + 0.213*cos(theta5) + 0.213*cos(theta5 + theta6))*sin(theta4) + 0.0955*cos(theta4),
        -1.0*(-1.0*fl_root_radius*sin(theta5 + theta6) - 0.213*sin(theta5) - 0.213*sin(theta5 + theta6))*cos(theta4),
        -1.0*(-1.0*fl_root_radius*sin(theta5 + theta6) - 0.213*sin(theta5 + theta6))*cos(theta4)]])
    
    #feet_linear velocities
    vfl = Jfl @ dqfl

    #-----RARE RIGHT LEG---------------------------------------------------------------------------------
    
    theta7 = rrt1
    theta8 = rrt2
    theta9 = rrt3
    rr_root_radius = rrr
    dqrr = np.array([rrd1, rrd2, rrd3])

    # position
    prr = np.array([[-1.0*rr_root_radius*sin(theta8 + theta9) - 0.213*sin(theta8) - 0.213*sin(theta8 + theta9) - 0.16783],
       [1.0*(1.0*rr_root_radius*cos(theta8 + theta9) + 0.213*cos(theta8) + 0.213*cos(theta8 + theta9))*sin(theta7) - 0.0955*cos(theta7) - 0.0465],
       [-1.0*(1.0*rr_root_radius*cos(theta8 + theta9) + 0.213*cos(theta8) + 0.213*cos(theta8 + theta9))*cos(theta7) - 0.0955*sin(theta7) - 0.04232]])

    #Jacobian
    Jrr = np.array([[0,
        -1.0*rr_root_radius*cos(theta8 + theta9) - 0.213*cos(theta8) - 0.213*cos(theta8 + theta9),
        -1.0*rr_root_radius*cos(theta8 + theta9) - 0.213*cos(theta8 + theta9)],
       [1.0*(1.0*rr_root_radius*cos(theta8 + theta9) + 0.213*cos(theta8) + 0.213*cos(theta8 + theta9))*cos(theta7) + 0.0955*sin(theta7),
        1.0*(-1.0*rr_root_radius*sin(theta8 + theta9) - 0.213*sin(theta8) - 0.213*sin(theta8 + theta9))*sin(theta7),
        1.0*(-1.0*rr_root_radius*sin(theta8 + theta9) - 0.213*sin(theta8 + theta9))*sin(theta7)],
       [1.0*(1.0*rr_root_radius*cos(theta8 + theta9) + 0.213*cos(theta8) + 0.213*cos(theta8 + theta9))*sin(theta7) - 0.0955*cos(theta7),
        -1.0*(-1.0*rr_root_radius*sin(theta8 + theta9) - 0.213*sin(theta8) - 0.213*sin(theta8 + theta9))*cos(theta7),
        -1.0*(-1.0*rr_root_radius*sin(theta8 + theta9) - 0.213*sin(theta8 + theta9))*cos(theta7)]])
    
    #feet_linear velocities
    vrr = Jrr @ dqrr


    #-----RARE LEFT LEG-----------------------------------------------------------------------------------
    
    theta10 = rlt1
    theta11 = rlt2
    theta12 = rlt3
    rl_root_radius = rlr
    dqrl = np.array([rld1, rld2, rld3])

    # position
    prl = np.array([[-1.0*rl_root_radius*sin(theta11 + theta12) - 0.213*sin(theta11) - 0.213*sin(theta11 + theta12) - 0.16783],
       [1.0*(1.0*rl_root_radius*cos(theta11 + theta12) + 0.213*cos(theta11) + 0.213*cos(theta11 + theta12))*sin(theta10) + 0.0955*cos(theta10) + 0.0465],
       [-1.0*(1.0*rl_root_radius*cos(theta11 + theta12) + 0.213*cos(theta11) + 0.213*cos(theta11 + theta12))*cos(theta10) + 0.0955*sin(theta10) - 0.04232]])
    #Jacobian
    Jrl = np.array([[0,
        -1.0*rl_root_radius*cos(theta11 + theta12) - 0.213*cos(theta11) - 0.213*cos(theta11 + theta12),
        -1.0*rl_root_radius*cos(theta11 + theta12) - 0.213*cos(theta11 + theta12)],
       [1.0*(1.0*rl_root_radius*cos(theta11 + theta12) + 0.213*cos(theta11) + 0.213*cos(theta11 + theta12))*cos(theta10) - 0.0955*sin(theta10),
        1.0*(-1.0*rl_root_radius*sin(theta11 + theta12) - 0.213*sin(theta11) - 0.213*sin(theta11 + theta12))*sin(theta10),
        1.0*(-1.0*rl_root_radius*sin(theta11 + theta12) - 0.213*sin(theta11 + theta12))*sin(theta10)],
       [1.0*(1.0*rl_root_radius*cos(theta11 + theta12) + 0.213*cos(theta11) + 0.213*cos(theta11 + theta12))*sin(theta10) + 0.0955*cos(theta10),
        -1.0*(-1.0*rl_root_radius*sin(theta11 + theta12) - 0.213*sin(theta11) - 0.213*sin(theta11 + theta12))*cos(theta10),
        -1.0*(-1.0*rl_root_radius*sin(theta11 + theta12) - 0.213*sin(theta11 + theta12))*cos(theta10)]])
    #feet_linear velocities
    vrl = Jrl @ dqrl

    return pfr, vfr, Jfr, \
           pfl, vfl, Jfl, \
           prr, vrr, Jrr, \
           prl, vrl, Jrl
           


if __name__ == '__main__':
    cc.compile()
