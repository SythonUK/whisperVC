import sys
import os
import argparse
import numpy as np
from matplotlib import pylab as plt
import glob
import time
import chainer
from chainer import cuda, Variable
from chainer import optimizers
from chainer import serializers
from highway import in2out_highway as Generator
import h5py
import cupy
import hts_mlpg

np.random.seed(2016)
cupy.random.seed(2016)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=200,
                        help="batch size")
    parser.add_argument('--epoch', type=int, default=25,
                        help="# of epochs for pre-training of generator")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--nhid', type=int, default=512,
                        help="# of hidden units")
    parser.add_argument('--hw', type=int, default=1,
                        help="highway flag")
    parser.add_argument('--omc', type=int, default=59,
                        help="order of mel-cepstral coefficients")
    parser.add_argument('outdir', type=str,
                        help="dirname to save outputs")
    parser.add_argument('src', type=str,
                        help="name of source speaker")
    parser.add_argument('tgt', type=str,
                        help="name of target speaker")
    return parser.parse_args()

# make dir if it is not exists
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

# zero mean, unit variance normarization
def normalize_gauss(z, mean, var, inv=False):
    if inv == False:
        return 1.0 / np.sqrt(var) * ( z - mean ) + 0
    else:
        return np.sqrt(var) / 1.0 * ( z - 0 ) + mean

# calculate R (paramerter generation matrix)
def calc_R(E, win):
    T = E.shape[0]
    mlpg = hts_mlpg.MLPG() # parameter generation class
    W = mlpg.construct_W(T, win) # delta matrix

    mlpg.set_param(E, W)
    R = mlpg.R

    return R, W

# train generator
def train(g, train_hdf5, stats_hdf5, gpu, batchsize, optimizer, hw=1):
    keys = train_hdf5.keys()
    n_train = len(keys)

    # training
    perm = np.random.permutation(keys)
    mean_loss_mc = 0.0
    omc = g.omc
    wins = [[1.0], [-0.5, 0.0, 0.5]]

    for i in range(n_train):
        key = perm[i]
        src_batch = train_hdf5[key]['src'].value
        tgt_batch = train_hdf5[key]['tgt'].value
        if stats_hdf5 is not None:
            src_batch = normalize_gauss(src_batch, stats_hdf5['src']['mc']['mean'], stats_hdf5['src']['mc']['var']).astype(np.float32)
            tgt_batch = normalize_gauss(tgt_batch, stats_hdf5['tgt']['mc']['mean'], stats_hdf5['tgt']['mc']['var']).astype(np.float32)
        R, W = calc_R(src_batch, wins)
        R, W = R.toarray(), W.toarray()
        if gpu >= 0:
            x = cuda.to_gpu(src_batch, gpu)
            y = cuda.to_gpu(tgt_batch[:, :omc], gpu)
            R = cuda.to_gpu(R, gpu)

        optimizer.target.cleargrads()
        loss, gen = g.forward(Variable(x), Variable(y), R, hw=hw)
        loss.backward()
        optimizer.update()

        mean_loss_mc += float(cuda.to_cpu(loss.data))
    return mean_loss_mc / n_train

def main():
    args = parse_args()
    mkdir(args.outdir)

    train_hdf5 = h5py.File(args.src + '_' + args.tgt + '.h5', 'r')
    stats_hdf5 = h5py.File('stats.h5', 'r')

    # model definition
    g = Generator(args.omc*2, args.nhid, args.omc*2)

    # optimizer
    optimizer_g = chainer.optimizers.AdaGrad(lr=args.lr) # g training
    optimizer_g.setup(g)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        g.to_gpu()

    # training
    training_loss_g = np.zeros(args.epoch)
    for epoch in range(1, args.epoch + 1):
        print('epoch %d:' % epoch)
        start = time.time()
        loss_train_g = train(
            g, train_hdf5, stats_hdf5,
            args.gpu, args.batch, optimizer_g, args.hw
        )
        end = time.time()
        print('training loss of g: %f' % loss_train_g)
        print('training time: %f' % (end-start))
        training_loss_g[epoch-1] = loss_train_g
        sys.stdout.flush()

        # Save the model and the optimizer
        serializers.save_hdf5(os.path.join(args.outdir, 'g.model'), g)
        serializers.save_hdf5(os.path.join(args.outdir, 'g.state'), optimizer_g)

    hdf5 = h5py.File(os.path.join(args.outdir, 'results.h5'), 'w')
    hdf5.create_group('train_loss_g')
    hdf5.create_dataset('train_loss_g/mgc', data=training_loss_g)

if __name__ == "__main__":
    sys.exit(main())
