import numpy as np
import pysptk
from scipy.io import wavfile
import pyworld
import argparse
import sys
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omc', type=int, default=59,
                        help="order of mel-cepstral coefficients")
    parser.add_argument('--shiftl', type=int, default=5,
                        help="frame shift [ms]")
    parser.add_argument('--fs', type=int, default=16000,
                        help="sampling rate [Hz]")
    parser.add_argument('src', type=str,
                        help='name of source speaker')
    parser.add_argument('minf0_s', type=int,
                        help='minimum value of F0 (src)')
    parser.add_argument('maxf0_s', type=int,
                        help='maximum value of F0 (src)')
    parser.add_argument('tgt', type=str,
                        help='name of target speaker')
    parser.add_argument('minf0_t', type=int,
                        help='minimum value of F0 (tgt)')
    parser.add_argument('maxf0_t', type=int,
                        help='maximum value of F0 (tgt)')
    return parser.parse_args()

# make dir if it is not exists
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def main():
    args = parse_args()
    fs = args.fs
    fftl = pyworld.get_cheaptrick_fft_size(fs)
    alpha = pysptk.util.mcepalpha(fs)
    hop_length = int(fs * args.shiftl * 0.001)

    spks = [args.src, args.tgt]
    minf0s = [args.minf0_s, args.minf0_t]
    maxf0s = [args.maxf0_s, args.maxf0_t]

    mkdir(os.path.join('data', 'mc'))
    mkdir(os.path.join('data', 'f0'))

    for spk, minf0, maxf0 in zip(spks, minf0s, maxf0s):
        mkdir(os.path.join('data', 'mc', spk))
        mkdir(os.path.join('data', 'f0', spk))
        fnames = glob.glob(os.path.join('data', 'wav', spk, '*.wav'))
        fnames.sort()
        for fn in fnames:
            base = os.path.basename(fn).split('.')[0]
            print('Processing %s (%s)' % (base, spk))
            sr, x = wavfile.read(fn)
            x = x.astype(np.float64)

            # extract F0
            f0, timeaxis = pyworld.dio(x, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=args.shiftl)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
            np.save(os.path.join('data', 'f0', spk, base), f0)

            # extract mel-cepstral coefficients
            sp = pyworld.cheaptrick(x, f0, timeaxis, fs, f0_floor=minf0, fft_size=fftl)
            mc = pysptk.sp2mc(sp, order=args.omc, alpha=alpha)
            np.save(os.path.join('data', 'mc', spk, base), mc)

if __name__ == '__main__':
    sys.exit(main())
