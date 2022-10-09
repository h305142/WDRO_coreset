
import gc
import sys
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from mosek.fusion import *
import random
import scipy.io as scio
import os
import copy
import time
import math

def uniform_sampling(X, y, samp_size):
    n, d = X.shape
    ind = np.random.permutation(n)[0:samp_size]
    X_u = X[ind, :]
    y_u = y[ind]
    w = n/samp_size
    w_u = np.ones(samp_size) * w
    return X_u, y_u, w_u


class DistributionallyHuber:
    def __init__(self, d, N,X,y,X_ts,y_ts,mod_sel,weight, set_delta):
        self.d, self.N = d, N
        self.set_gamma = 7
        self.weight = weight
        self.set_delta = set_delta
        self.set_dualnorm = 2
        if mod_sel==0:
            self.M = self.huber(X, y)
        elif mod_sel==1:
            self.M = self.wdroHuber(X, y)
            self.gamma = self.M.getParameter('gamma')
            self.gamma.setValue(self.set_gamma)

        self.beta = self.M.getVariable('beta')
        self.b=self.M.getVariable('b')

        self.theta = self.M.getParameter('WasRadius')
        self.delta = self.M.getParameter('delta')
        self.delta.setValue(self.set_delta)

        self.sol_time = []
        self.data_x=X
        self.data_y=y
        self.X_ts=X_ts
        self.y_ts=y_ts

    def wdroHuber(self, X, y):
        n, d = int(X.shape[0]), int(X.shape[1])  # num samples, dimension
        M = Model()
        # M.setSolverParam("optimizer", "conic")
        M.setLogHandler(open('result/huber/wdrohuber.txt', 'wt'))

        beta = M.variable('beta', d)
        b = M.variable('b')
        z = M.variable('z', n)
        r = M.variable('r')  # max (|\beta|_2, 2/\gamma)
        s = M.variable('s', n)  # s_i = |wx+b - (y+z)|
        t = M.variable('t', 1)  # t >= \sum z_i^2

        delta = M.parameter('delta')
        theta = M.parameter('WasRadius')
        gamma = M.parameter('gamma', 1)
        ones_p = M.parameter('ones_p', n)

        ones_p.setValue([1 for i in range(n)])

        n_inv = 1 / self.N
        weight1 = self.weight * n_inv
        weight_sqrt = np.sqrt(self.weight)
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                # Expr.mul(Expr.mul(theta, delta), r),
                Expr.mul(delta, Expr.mul(theta, r)),
                Expr.add(
                    Expr.mul(delta, Expr.sum(Expr.mulElm(s, weight1))),
                    Expr.mul(t, n_inv / 2)
                ),
            )

        )
        # r >= |\beta|_2, r >= sqrt(beta_1^2+...+beta_n^2)
        M.constraint(Expr.vstack(r, beta), Domain.inQCone())
        # r >= 2 / \gamma,   2 gamma r >= 2^2
        M.constraint(Expr.vstack(gamma, r, 2.0), Domain.inRotatedQCone())
        # s + (wx+b - (y+z)) >=0; s - (wx+b - (y+z)) >=0
        M.constraint(
            Expr.add(s,
                     Expr.sub(
                         Expr.add(Expr.mul(X, beta), Var.repeat(b, n)),
                         Expr.add(y, z))
                     ),
            Domain.greaterThan(0.0))
        M.constraint(
            Expr.sub(s,
                     Expr.sub(
                         Expr.add(Expr.mul(X, beta), Var.repeat(b, n)),
                         Expr.add(y, z))
                     ),
            Domain.greaterThan(0.0))
        # t >= \sum (\sqrt(w_i)*z_i)^2
        M.constraint(Expr.vstack(0.5, t, Expr.mulElm(z, weight_sqrt.squeeze())), Domain.inRotatedQCone())
        return M

    def huber(self, X, y):
        n, d = int(X.shape[0]), int(X.shape[1])  # num samples, dimension
        M = Model()
        # M.setSolverParam("optimizer", "conic")
        M.setLogHandler(open('result/huber/huber.txt', 'wt'))

        beta = M.variable('beta', d)
        b = M.variable('b')
        z = M.variable('z', n)
        s = M.variable('s', n)  # s_i = |wx+b - (y+z)|
        t = M.variable('t', 1)  # t >= \sum z_i'^2

        theta = M.parameter('WasRadius')
        delta = M.parameter('delta', 1)
        ones_p = M.parameter('ones_p', n)
        ones_p.setValue([1 for i in range(n)])

        n_inv = 1 / self.N
        weight1 = self.weight * n_inv
        weight_sqrt = np.sqrt(self.weight)
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                # Expr.sum(Expr.mul(delta, Expr.mulElm(s, weight1))),
                Expr.mul(delta, Expr.sum(Expr.mulElm(s, weight1))),
                Expr.mul(t, n_inv / 2)
            )
        )
        # s + (wx+b - (y+z)) >=0; s - (wx+b - (y+z)) >=0
        M.constraint(
            Expr.add(s,
                     Expr.sub(
                         Expr.add(Expr.mul(X, beta), Var.repeat(b, n)),
                         Expr.add(y, z))
                     ),
            Domain.greaterThan(0.0))
        M.constraint(
            Expr.sub(s,
                     Expr.sub(
                         Expr.add(Expr.mul(X, beta), Var.repeat(b, n)),
                         Expr.add(y, z))
                     ),
            Domain.greaterThan(0.0))

        # t >= \sum (\sqrt(w_i)*z_i)^2
        M.constraint(Expr.vstack(0.5, t, Expr.mulElm(z, weight_sqrt.squeeze())), Domain.inRotatedQCone())
        return M

    def iter_data(self):

        return  self.simulate()

    def iter_radius(self, epsilon_range):

        for epsilon in epsilon_range:
            yield self.solve(epsilon)

    def simulate(self):
        pass

    def solve(self, epsilon):
        pass


class SimSet1(DistributionallyHuber):
    def __init__(self,d, N, eps_range,X,y,X_ts,y_ts,mod_sel,weight, set_delta):
        self.theta_range = eps_range
        super().__init__(d, N, X, y, X_ts, y_ts, mod_sel, weight, set_delta)

    def simulate(self):

        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = zip(
            *[(_w, _p, _r, train_loss, train_predict, test_loss, test_predict)
              for _w, _p, _r, train_loss, train_predict, test_loss, test_predict in
              self.iter_radius(self.theta_range)])
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict

    def solve(self, epsilon):
        self.theta.setValue(epsilon)
        self.set_delta
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        out_perf=0
        beta_sol = self.beta.level()
        b_val=self.b.level()
        train_loss = self.huber_loss(self.data_x, self.data_y, beta_sol,b_val,self.set_delta)
        train_l2 = self.l2_loss(self.data_x, self.data_y, beta_sol,b_val)
        test_loss = self.huber_loss(self.X_ts, self.y_ts, beta_sol,b_val,self.set_delta)
        test_l2 = self.l2_loss(self.X_ts, self.y_ts, beta_sol,b_val)
        return beta_sol, out_perf,self.M.primalObjValue(), train_loss, train_l2, test_loss, test_l2

    def run_sim(self):

        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = self.iter_data()
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict

    def huber_loss(self, X, y, beta, b_val, delta):
        y_pred = np.dot(X, beta) + b_val
        z = np.absolute(y_pred - y.squeeze())
        huber_loss_vec = np.zeros(z.shape)
        idx_l = np.where(z > delta)[0]
        idx_s = np.where(z <= delta)[0]
        print('num > delta: ', idx_l.shape)
        huber_loss_vec[idx_l] = delta * (z[idx_l] - 0.5 * delta)
        huber_loss_vec[idx_s] = (z[idx_s] ** 2) * 0.5
        loss_h = np.sum(huber_loss_vec)
        return loss_h

    def l2_loss(self, X, y, beta, b_val):
        y_pred = np.dot(X, beta) + b_val
        z = y_pred - y.squeeze()
        l2_loss = np.sum(z**2)
        return l2_loss