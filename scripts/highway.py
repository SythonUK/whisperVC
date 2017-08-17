import numpy as np
import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from parameter_generation import parameter_generation
import cupy

class WeightedSum(chainer.Function):
    def forward_cpu(self, inputs):
        x, wgt, y = inputs
        return x + wgt * y,

    def forward_gpu(self, inputs):
        x, wgt, y = inputs
        return x + wgt * y,

    def backward_cpu(self, inputs, grad_outputs):
        x, wgt, y = inputs
        gz, = grad_outputs

        gx = gz
        gw = y * gz
        gy = wgt * gz

        return gx, gw, gy,

    def backward_gpu(self, inputs, grad_outputs):
        x, wgt, y = inputs
        gz, = grad_outputs

        gx = gz
        gw = y * gz
        gy = wgt * gz

        return gx, gw, gy,

def weighted_sum(x, w, y):
    return WeightedSum()(x, w, y)

class in2out_highway(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(in2out_highway, self).__init__(
            lin   = L.Linear(n_in, n_units),
            lh1   = L.Linear(n_units, n_units),
            lh2   = L.Linear(n_units, n_units),
            lout  = L.Linear(n_units, n_out),
            lhw   = L.Linear(n_in/2, n_out/2),
        )
        self.omc = n_in / 2

    def __call__(self, x, R, hw=1):
        h = F.dropout( F.relu( self.lin(x) ))
        h = F.dropout( F.relu( self.lh1(h) ))
        h = F.dropout( F.relu( self.lh2(h) ))
        h = self.lout(h)
        h = parameter_generation(R, h)
        if hw == 1:
            x_s, x_d = F.split_axis(x, [self.omc], 1)
            T = F.sigmoid( self.lhw(x_s) )
            return weighted_sum(x_s, T, h)
        return h

    def forward(self, x, y, R, hw=1):
        gen = self(x, R, hw=hw)
        loss = F.mean_squared_error(y, gen)
        return loss, gen
