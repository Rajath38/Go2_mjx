from numba.pycc import CC
from go2_deploy.utils import math_function as MF 
import numpy as np

cc = CC('Go2_orientation_CF')


@cc.export('run', '(f8[:,:], f8[:],'
                  'f8[:], f8[:], f8, f8, f8)')
def run(R0, w0,
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


if __name__ == '__main__':
    cc.compile()