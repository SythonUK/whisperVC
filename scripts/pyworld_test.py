import sys, os
import subprocess
import numpy as np
import pyworld
import pysptk
from scipy.io import wavfile
import glob

def main():
    shiftl = 5
    fftl = 1024
    n_test = 10
    minf0 = float(sys.argv[1])
    maxf0 = float(sys.argv[2])
    spk = sys.argv[3]
    fnames = glob.glob(os.path.join('data', 'wav', spk, '*.wav'))
    fnames.sort()

    for fn in fnames[-n_test:]:
        sr, x = wavfile.read(fn)
        x = x.astype(np.float64)

        # extract F0
        f0, timeaxis = pyworld.dio(x, sr, f0_floor=minf0, f0_ceil=maxf0, frame_period=shiftl)
        f0 = pyworld.stonemask(x, f0, timeaxis, sr)

        # extract aperiodicity
        ap = pyworld.d4c(x, f0, timeaxis, sr, fft_size=fftl)

        # extract mel-cepstral coefficients
        sp = pyworld.cheaptrick(x, f0, timeaxis, sr, f0_floor=minf0, fft_size=fftl)

        x = pyworld.synthesize(f0, sp, ap, sr, shiftl)
        wavfile.write('gomi.wav', sr, x.astype(np.int16))
        cmd = 'aplay gomi.wav'
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    sys.exit(main())
