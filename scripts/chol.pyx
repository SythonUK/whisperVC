#################################################################
#                           chol.pyx                            #
#################################################################
#           Copyright (c) 2016 Shinnosuke Takamichi             #
#      This software is released under the MIT License.         #
#       http://opensource.org/licenses/mit-license.php          #
#################################################################

import numpy as np

def cholesky(R, width=3, max_L=3):
   T = R.shape[0]

   R[0, 0] = np.sqrt(R[0, 0])
   for j in range(1, width):
      R[0, j] /= R[0, 0]
   for t in range(1, T):
      for j in range(1, width):
         if t - j >= 0:
            R[t, 0] -= R[t - j, j] * R[t - j, j]
         R[t, 0] = np.sqrt(R[t, 0])

   for j in range(1, width):
      for k in range(0, max_L):
         if j != width - 1 :
            R[t, j] -= R[t - k - 1, j - k] * \
             R[t - k - 1, j + 1]
         R[t, j] /= R[t, 0]
   return R

def chol_inv(R, width=3):
   # r = np.identity(len(R))
   # P = my_fb_mat(R, r, width)
   P = calc_P(R, width)
   return P

def calc_P(R, w):
   T = R.shape[0]
   g = np.zeros(R.shape)
   g[0, 0] = 1.0 / R[0, 0]
   hold = np.zeros(T)
   P = np.zeros(R.shape)

   for t in range(1, T):
      hold *= 0.0
      for j in range(1, w):
         if (t - j >= 0) and (R[t, t - j] != 0.0):
            hold[0:t+1] += R[t, t - j] * g[t - j, 0:t+1]
      hold[t] -= 1.0
      g[t, 0:t+1] = - hold[0:t+1] / R[t, t]

   P[T - 1, :] = g[T - 1, :] / R[T - 1, T - 1]
   R = R.T

   for t in range(T - 2, -1, -1):
      hold *= 0.0
      for j in range(1, w):
         if (t + j < T) and (R[t, t + j] != 0.0):
            hold += R[t, t + j] * P[t + j, :]
      P[t, :] = (g[t, :] - hold) / R[t, t]

   return P

def my_fb_mat(R, r, w):
   T = R.shape[0]
   g = np.zeros((T, T))
   g[0, :] = r[0, :] / R[0][0]
   hold = np.zeros(len(r))
   P = np.zeros((T, T))


   # forward
   for t in range(1, T):
      hold *= 0.0
      for j in range(1, w):
         if (t - j >= 0) and (R[t][t - j] != 0.0):
            hold += R[t][t - j] * g[t - j, :]
      g[t, :] = (r[t, :] - hold) / R[t][t]

   # backward
   P[T - 1, :] = g[T - 1, :] / R[T - 1][T - 1]
   Ti = range(0, T - 1)
   Ti.reverse()
   R = R.T

   for t in Ti:
      hold *= 0.0
      for j in range(1, w):
         if (t + j < T) and (R[t][t + j] != 0.0):
            hold += R[t][t + j] * P[t + j, :]
      P[t, :] = (g[t, :] - hold) / R[t][t]

   return P
   # print g, hold, r[t, :], R[t][t]
   # exit()

def forward(R, r, g, w):
   T = len(r)
   g[0] = r[0] / R[0][0]

   for t in range(1, T):
      hold = 0.0
      for j in range(1, w):
         if (t - j >= 0) and (R[t - j][j] != 0.0):
            hold += R[t - j][j] * g[t - j]
      g[t] = (r[t] - hold) / R[t][0]

def backward(R, r, g, w, c):
   w = 2
   T = len(r)
   c[T - 1] = g[T - 1] / R[T - 1][0]
   Ti = range(0, T - 1)
   Ti.reverse()
   for t in Ti:
      hold = 0.0
      for j in range(1, w):
         if (t + j < T) and (R[t][j] != 0.0):
            hold += R[t][j] * c[t + j]
      c[t] = float(g[t] - hold) / R[t][0]
