
import scipy
from sklearn import mixture
from scipy import linalg
from scipy import sparse
import chol
import numpy as np
#import time

class MLPG():
  def set_param(self, E, W):
    self.W = W
    self.E = E
    self.P = self.construct_P()
    self.R = self.construct_R()

  def construct_P(self):
    R = self.W.T.dot(self.W)
    L = scipy.linalg.cholesky(R.todense(), check_finite=False, lower=True)
    P = scipy.sparse.lil_matrix(np.around(chol.chol_inv(L), 20))
    return P

  def construct_R(self):
    return self.P.dot(self.W.T)

  def mlpg(self):
    self.cq = self.R.dot(self.E)
    # cq = P * r = (W^T * D^-1 * W)^-1 * W^T * D^-1 * E

  def construct_W(self, T, win):
    nw= len(win)
    W = scipy.sparse.lil_matrix((nw * T, T))
    for t in range(T):
      for n in range(len(win)):
        for i in range(len(win[n])):
          pos = t + i - (len(win[n]) - 1) /2
          if pos >= 0 and pos < T:
            W[nw * t + n, pos] = win[n][i]
    return W

  def get_trjprob(self, c):
    T = c.shape[0]
    trjgmm = mixture.GMM(n_components = 1, covariance_type = 'full')
    trjgmm.means_ = np.zeros((1, T))
    trjgmm.covars_= np.zeros((1, T, T))
    trjgmm.means_[0, :] = self.cq
    trjgmm.covars_[0, :, :]= self.P.todense()
    return trjgmm.score([c])[0] / float(T)
