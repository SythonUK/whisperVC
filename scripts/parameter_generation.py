import numpy as np
import cupy
from chainer import function, cuda
from chainer.utils import type_check

class ParameterGeneration(function.Function):
    """
    Parameter generation based on
    Maximum-Likelihood criterion
    """

    def __init__(self, R, n_win=2):
        self.R = R
        self.n_win = n_win

    def forward_cpu(self, inputs):
        O, = inputs
        T, dim = O.shape[0], O.shape[1] / self.n_win
        y = np.zeros((T, dim), dtype=np.float32)
        for d in xrange(dim):
            O_d = O[:, d::dim].flatten()
            y[:, d] = self.R.dot(O_d)
        return y,

    def forward_gpu(self, inputs):
        O, = inputs
        T, dim = O.shape[0], O.shape[1] / self.n_win
        y = cupy.zeros((T, dim), dtype=cupy.float32)
        for d in xrange(dim):
            O_d = O[:, d::dim].reshape(1, T*self.n_win)
            y_d = O_d.dot(self.R.T).T
            y[:, d] = y_d.flatten()
        return y,

    def backward_cpu(self, inputs, go):
        O, = inputs
        gz, = go
        [T, dim] = gz.shape
        gx = np.zeros((T, O.shape[1]), dtype=np.float32)
        for d in xrange(dim):
            gz_d = gz[:, d].reshape(T, 1)
            gx[:, d::dim] = self.R.T.dot(gz_d).reshape(T, self.n_win)
        return gx,

    def backward_gpu(self, inputs, go):
        O, = inputs
        gz, = go
        [T, dim] = gz.shape
        gx = cupy.zeros((T, O.shape[1]), dtype=cupy.float32)
        for d in xrange(dim):
            gz_d = gz[:, d].reshape(T, 1)
            gx[:, d::dim] = self.R.T.dot(gz_d).reshape(T, self.n_win)
        return gx,

def parameter_generation(R, O):
    """
    Parameter Generation.
     R: constant matrix for parameter generation
     O: predicted parameters (including dynamic features: delda, delta-delta)
    """
    return ParameterGeneration(R)(O)
