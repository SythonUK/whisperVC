import sys
import os
import argparse
import numpy as np
import glob
import time
import chainer
from chainer import cuda, Variable
from chainer import serializers
from highway import in2out_highway as Generator
import h5py
import cupy
from scipy.io import wavfile
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
import hts_mlpg
import pyworld
import subprocess

np.random.seed(2016)
cupy.random.seed(2016)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhid', type=int, default=512,
                        help="# of hidden units")
    parser.add_argument('--hw', type=int, default=1,
                        help="highway flag")
    parser.add_argument('--omc', type=int, default=59,
                        help="order of mel-cepstral coefficients")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument('--shiftl', type=int, default=5,
                        help="frame shift [ms]")
    parser.add_argument('--fs', type=int, default=16000,
                        help="sampling rate [Hz]")
    parser.add_argument('--VUF', type=int, default=1,
                        help="VUF for whisper filtering (if 0, no filtering)")
    parser.add_argument('gendir', type=str,
                        help="dirname to load trained generator model")
    parser.add_argument('ipt', type=str,
                        help="input wav file")
    parser.add_argument('minf0_s', type=int,
                        help='minimum value of F0 (src)')
    parser.add_argument('maxf0_s', type=int,
                        help='maximum value of F0 (src)')
    parser.add_argument('opt', type=str,
                        help="output wav")
    return parser.parse_args()

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

# zero mean, unit variance normarization
def normalize_gauss(z, mean, var, inv=False):
    if inv == False:
        return 1.0 / np.sqrt(var) * ( z - mean ) + 0
    else:
        return np.sqrt(var) / 1.0 * ( z - 0 ) + mean

# calculate R (parameter generation matrix)
def calc_R(E, win):
    T = E.shape[0]
    mlpg = hts_mlpg.MLPG() # parameter generation class
    W = mlpg.construct_W(T, win) # delta matrix

    mlpg.set_param(E, W)
    R = mlpg.R

    return R, W

# generate mc
def generate(g, mc, stats_hdf5, gpu, wins, hw=1):
    omc = g.omc
    if stats_hdf5 is not None:
        mc = normalize_gauss(mc, stats_hdf5['src']['mc']['mean'], stats_hdf5['src']['mc']['var']).astype(np.float32)
    R, W = calc_R(mc, wins)
    R, W = R.toarray(), W.toarray()
    if gpu >= 0:
        x = cuda.to_gpu(mc, gpu)
        R = cuda.to_gpu(R, gpu)

    with chainer.using_config('train', False):
        gen = g(Variable(x), R, hw=hw)

    gen = cuda.to_cpu(gen.data)

    if stats_hdf5 is not None:
        gen = normalize_gauss(gen, stats_hdf5['tgt']['mc']['mean'][:omc], stats_hdf5['tgt']['mc']['var'][:omc], inv=True)
    return gen

def main():
    args = parse_args()
    fs = args.fs
    fftl = pyworld.get_cheaptrick_fft_size(fs)
    alpha = pysptk.util.mcepalpha(fs)
    stats_hdf5 = h5py.File('stats.h5', 'r')
    wins = [[1.0], [-0.5, 0.0, 0.5]]
    hop_length = int(16000 * args.shiftl * 0.001)

    # model definition
    g = Generator(args.omc*2, args.nhid, args.omc*2)
    serializers.load_hdf5(os.path.join(args.gendir, 'g.model'), g)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        g.to_gpu()

    # read wav
    sr, x = wavfile.read(args.ipt)
    x = x.astype(np.float64)

    # extract F0
    f0, timeaxis = pyworld.dio(x, fs, f0_floor=args.minf0_s, f0_ceil=args.maxf0_s, frame_period=args.shiftl)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)

    # extract aperiodicity
    ap = pyworld.d4c(x, f0, timeaxis, fs, fft_size=fftl)

    # extract mel-cepstral coefficients
    sp = pyworld.cheaptrick(x, f0, timeaxis, fs, f0_floor=args.minf0_s, fft_size=fftl)
    mc = pysptk.sp2mc(sp, order=args.omc, alpha=alpha)

    # convert F0 (linear transformation)
    mean_src = stats_hdf5['src']['f0']['mean']
    var_src = np.sqrt(stats_hdf5['src']['f0']['var'])
    mean_tgt = stats_hdf5['tgt']['f0']['mean']
    var_tgt = np.sqrt(stats_hdf5['tgt']['f0']['var'])
    std = np.sqrt(var_tgt / var_src)
    f0[f0 != 0.0] = (f0[f0 != 0.0] - mean_src) * std + mean_tgt

    x = pyworld.synthesize(f0, sp, ap, fs, args.shiftl)
    x_syn = pyworld.synthesize(f0, sp, ap, fs, args.shiftl)

    # convert mel-cepstral coefficients
    MC = apply_delta_win(mc[:, 1:], wins)
    gen = generate(g, MC, stats_hdf5, args.gpu, wins, args.hw)
    mc[:, 1:] = gen - mc[:, 1:]
    mc = mc.astype(np.float64)
    mc0 = np.copy(mc[:, 0])

    # synthesize waveform
    mc[:, 0] = 0
    engine = Synthesizer(MLSADF(order=args.omc, alpha=alpha), hopsize=hop_length)
    b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
    x_syn = engine.synthesis(x, b)

    # whisper filtering
    if args.VUF > 0:
        mc[:, 0] = mc0
        sp = pysptk.mc2sp(mc, alpha=alpha, fftlen=fftl)
        num = int(args.VUF * fftl / (fs * 0.001))
        T = len(sp)
        f0 = np.zeros(T, dtype=np.float64)
        w = np.linspace(0.01, np.exp(1), num=num)
        wsp = np.ones(sp.shape[1]) * np.exp(1)
        wsp[:num] = w
        sp_w = np.exp(np.log(sp) + np.log(wsp))
        x_syn = pyworld.synthesize(f0, sp_w, ap, fs, args.shiftl)

    # output wav file
    wavfile.write(args.opt, sr, x_syn.astype(np.int16))

    # whisper from rightside
    sox = 'sox ' + args.opt + ' gomi.wav remix 1v0.0 1v0.5'
    subprocess.call(sox, shell=True)
    mv = 'mv gomi.wav ' + args.opt
    subprocess.call(mv, shell=True)

if __name__ == "__main__":
    sys.exit(main())
