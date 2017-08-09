
import hts_mlpg # ML-based parameter generation
import numpy as np
import scipy
from scipy import sparse # for W
from scipy.linalg import block_diag


def hts_read_win(fn):
    u"""read HTS-defined window files

    Attributes:
    fn: window filename
    """
    for line in open(fn, 'r'):
        win = line.rstrip('\n').split(' ')
        break
    return win[1:]

def mlpg_from_pdf(E, win):
    u"""ML-based parameter generation

    Attributes:
        E: mean vector sequence (dim x T)
        win: window functions
    """
    T = E.shape[0]
    mlpg = hts_mlpg.MLPG() # parameter generation class
    W = mlpg.construct_W(T, win) # delta matrix

    mlpg.set_param(E, W)
    R = mlpg.R

    return R, W

if __name__ == '__main__':
    # statistics (1 dim * 3 wins, 3 frames)
    means = np.array([[1.0,1.0,1.0],[2.0,-1.0,-1.0],[1.0,1.0,1.0]], dtype=float)
    invvars = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]], dtype=float)

    # window functions
    win = [ hts_read_win("win/mgc.win" + str(i)) for i in range(1, 4) ]
    print win

    # parameter generation (1 dim, 3 frames)
    y = mlpg_from_pdf(means, win)
    print y
