#################################################################
#                      make_paradata.py                         #
#################################################################
#     Copyright (c) 2017 Shinnosuke Takamichi & Yuki Saito      #
#      This software is released under the MIT License.         #
#       http://opensource.org/licenses/mit-license.php          #
#################################################################

import numpy as np
import argparse
import sys
import os
import h5py
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omc', type=int, default=59,
                        help="order of mel-cepstral coefficients")
    parser.add_argument('src', type=str,
                        help='name of source speaker')
    parser.add_argument('tgt', type=str,
                        help='name of target speaker')
    return parser.parse_args()

# distance measures for DTW
class my_distance_measure():
    def __init__(self, idx = None):
        self.idx = idx

    # euclidean distance
    def euclid(self, x, y):
        index = range(x.shape[0]) if (self.idx is None) else self.idx
        return np.sum(np.square(x[index] - y[index]), axis = len(x.shape) - 1)

    # mel-cepstral distortion
    def melcep_dist(self, x, y):
        index = range(len(x)) if (self.idx is None) else self.idx
        return 10.0 / np.log(10.0) * np.sqrt(2.0 * self.euclid(x, y))

# dynamic time warping
def dtw(x, y, dist, wgt = [[1.0, 1.0], [1.0, 1.0]]):
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    w = np.array(wgt)

    # frame-wise calculation
    I = np.ones((c, len(x[0])))
    for i in range(r):
       D1[i, :] = dist(I * x[i], y)

    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(w[1, 1] * D0[i, j], w[0, 1] * D0[i, j+1], w[1, 0] * D0[i+1, j])

    if len(x)==1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0, wgt)
    return path

def _traceback(D, wgt):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    w = np.array(wgt)

    while ((i > 0) or (j > 0)):
        tb = np.argmin((w[1, 1] * D[i, j], w[0, 1] * D[i, j+1], w[1, 0] * D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

# calculate dynamic feaures using wins
def apply_delta_win(x, wins):
    o = [np.zeros(np.array(x).shape) for i in range(len(wins))]

    for win in range(len(wins)):
        if len(wins) % 2 != 1:
            ValueError("win length must be odd.")

        for w in range(len(wins[win])):
            shift_t = int(len(wins[win]) * 0.5) - w
            shift_x = np.roll(x, shift_t, axis = 0)

            if shift_t > 0:
                shift_x[:shift_t, :] = 0.0
            elif shift_t < 0:
                shift_x[shift_t:, :] = 0.0

            o[win] += wins[win][w] * shift_x

    # concatenate
    X = o[0]
    for win in range(1, len(wins)):
        X = np.c_[X, o[win]]

    return X

# estimate threshold values for cutting silent intervals
def est_thvals(fnames, src, tgt):
    minpows_s = []
    minpows_t = []
    for fn in fnames:
        base = os.path.basename(fn).split('.')[0]
        mc_s = np.load(os.path.join('data', 'mc', src, base) + '.npy')
        mc_t = np.load(os.path.join('data', 'mc', tgt, base) + '.npy')
        pow_src, pow_tgt = mc_s[:, 0], mc_t[:, 0]
        minpows_s.append(min(pow_src))
        minpows_t.append(min(pow_tgt))
    thval_s = max(minpows_s)
    thval_t = max(minpows_t)
    return thval_s, thval_t

def main():
    args = parse_args()
    dist = my_distance_measure()

    mcs_s = np.zeros(args.omc*2, dtype=np.float64)
    mcs_t = np.zeros(args.omc*2, dtype=np.float64)

    f0s_s = np.zeros(1, dtype=np.float64)
    f0s_t = np.zeros(1, dtype=np.float64)
    wins = [[1.0], [-0.5, 0.0, 0.5]]

    # make parallel data
    hdf5 = h5py.File(args.src + '_' + args.tgt + '.h5', 'w')
    fnames = glob.glob(os.path.join('data', 'wav', args.src, '*.wav'))
    fnames.sort()
    thval_s, thval_t = est_thvals(fnames, args.src, args.tgt)
    print(thval_s, thval_t)
    for fn in fnames:
        base = os.path.basename(fn).split('.')[0]
        # read data
        f0_s = np.load(os.path.join('data', 'f0', args.src, base) + '.npy')
        f0_t = np.load(os.path.join('data', 'f0', args.tgt, base) + '.npy')
        f0s_s = np.hstack((f0s_s, f0_s[f0_s != 0.0]))
        f0s_t = np.hstack((f0s_t, f0_t[f0_t != 0.0]))

        mc_s = np.load(os.path.join('data', 'mc', args.src, base) + '.npy')
        mc_t = np.load(os.path.join('data', 'mc', args.tgt, base) + '.npy')
        pow_src, pow_tgt = mc_s[:, 0], mc_t[:, 0]

        SRC = apply_delta_win(mc_s[:, 1:], wins)
        TGT = apply_delta_win(mc_t[:, 1:], wins)

        # remove sil
        SRC = SRC[np.where(pow_src >= thval_s)]
        TGT = TGT[np.where(pow_tgt >= thval_t)]

        # dtw and store
        path_src, path_tgt = dtw(SRC, TGT, dist.melcep_dist)

        mcs_s = np.vstack((mcs_s, SRC[path_src]))
        mcs_t = np.vstack((mcs_t, TGT[path_tgt]))

        hdf5.create_group(base)
        hdf5.create_dataset(base + '/src', data=SRC[path_src])
        hdf5.create_dataset(base + '/tgt', data=TGT[path_tgt])
        print(' %s[%d], %s[%d] -> [%d]' % (base + '(' + args.src + ')', len(SRC), base + '(' + args.tgt + ')', len(TGT), len(path_src)))
        sys.stdout.flush()
    hdf5.flush()
    hdf5.close()

    # calculate statistics
    mcs_s = mcs_s[1:]
    mcs_t = mcs_t[1:]
    mcs_s = mcs_s[1:]
    mcs_t = mcs_t[1:]

    f0s_s = f0s_s[1:]
    f0s_t = f0s_t[1:]

    hdf5 = h5py.File('stats.h5', 'w')
    hdf5.create_group('src')
    hdf5.create_group('src/mc')
    hdf5.create_dataset('src/mc/mean', data=np.mean(mcs_s, axis=0))
    hdf5.create_dataset('src/mc/var', data=np.var(mcs_s, axis=0))
    hdf5.create_group('src/f0')
    hdf5.create_dataset('src/f0/mean', data=np.mean(f0s_s))
    hdf5.create_dataset('src/f0/var', data=np.var(f0s_s))
    hdf5.create_group('tgt')
    hdf5.create_group('tgt/mc')
    hdf5.create_dataset('tgt/mc/mean', data=np.mean(mcs_t, axis=0))
    hdf5.create_dataset('tgt/mc/var', data=np.var(mcs_t, axis=0))
    hdf5.create_group('tgt/f0')
    hdf5.create_dataset('tgt/f0/mean', data=np.mean(f0s_t))
    hdf5.create_dataset('tgt/f0/var', data=np.var(f0s_t))
    hdf5.flush()
    hdf5.close()

if __name__ == '__main__':
    sys.exit(main())
