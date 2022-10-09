import numpy as np
from mosek.fusion import *
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

def svm_query_h(X, y, beta, b, lamb, gamma):
    signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
    start1=time.time()
    z1 = (np.dot(X, beta) - b)*signs
    # print('xbeta',time.time()-start1)
    z1 = z1[:, np.newaxis]
    z1_prime = 1 - z1
    z2_prime = 1 + z1
    z1_prime[z1_prime < 0] = 0
    z2_prime[z2_prime < 0] = 0
    # print(time.time() - start1)
    f1 = z1_prime
    f2 = z2_prime - lamb*gamma
    # print(time.time() - start1)
    F = np.hstack((f1, f2))
    return np.max(F, axis=1)


class DistributionallyRobustSVM:
    def __init__(self, d, N,X,y,X_ts,y_ts,mod_sel,weight,set_reg_coef):
        self.d, self.N = d, N
        self.gamma=7
        self.weight=weight
        self.set_length=1
        self.set_dualnorm=2
        self.set_reg_coef=set_reg_coef
        if mod_sel==0:
            self.M = self.primal_svm(X, y)
        elif mod_sel==1:
            self.M = self.wdrosvm_space(X, y)
        else:
            self.M = self.wdrosvm_hypercube(X, y)

        self.beta = self.M.getVariable('beta')
        self.b=self.M.getVariable('b')
        self.theta = self.M.getParameter('WasRadius')
        self.lamb=self.M.getVariable('Lambda')
        self.sol_time = []
        self.data_x=X
        self.data_y=y
        self.X_ts=X_ts
        self.y_ts=y_ts


    def pnorm(self,M, t, x, p):
        n = int(x.getSize())
        r = M.variable(n)
        M.constraint(Expr.sub(t, Expr.sum(r)), Domain.equalsTo(0.0))
        M.constraint(Expr.hstack(Var.repeat(t, n), r, x), Domain.inPPowerCone(1.0 - 1.0 / p))

    def softplus(self,M, t, u):
        n = t.getShape()[0]
        z1 = M.variable(n)
        z2 = M.variable(n)
        M.constraint(Expr.add(z1, z2), Domain.equalsTo(1))
        M.constraint(Expr.hstack(z1, Expr.constTerm(n, 1.0), Expr.sub(u, t)), Domain.inPExpCone())
        M.constraint(Expr.hstack(z2, Expr.constTerm(n, 1.0), Expr.neg(t)), Domain.inPExpCone())

    def wdrosvm_hypercube(self,X, y):
        n, d = int(X.shape[0]), int(X.shape[1])
        # n, d = int(X.shape[0]), int(X.shape[1])  # num samples, dimension
        M = Model()
        M.setSolverParam("optimizer", "conic")
        M.setLogHandler(open('result/WDROSVM.txt', 'wt'))

        beta = M.variable('beta', d)
        lamb = M.variable('Lambda')
        pp = M.variable('pp', [X.shape[0], d])
        pn = M.variable('pn', [X.shape[0], d])
        zp = M.variable('zp', [X.shape[0], d], Domain.greaterThan(0.))
        zn = M.variable('zn', [X.shape[0], d], Domain.greaterThan(0.))
        s = M.variable('s_i', X.shape[0], Domain.greaterThan(0.))
        # t = M.variable('t', 1, Domain.equalsTo(0.0))
        t = M.variable('t', 1, Domain.unbounded())
        b = M.variable('b', 1)

        theta = M.parameter('WasRadius')
        L = M.parameter('CubicLength')
        gamma = M.parameter('gamma')
        L.setValue(self.set_length)
        gamma.setValue(7)
        signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        y = np.diagflat(signs)
        p_norm = self.set_dualnorm
        n_inv = 1 / self.N

        # min t + (1/n * \sum s + lamb*theta)
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                t,
                Expr.add(
                   Expr.mul(Expr.sum(Expr.mulElm(s,self.weight)), n_inv),
                    Expr.mul(lamb, theta)
                )
            )
        )

        if p_norm == math.inf:
            M.constraint(Expr.sub(Expr.repeat(lamb, d, 0), beta), Domain.greaterThan(0.0))
            M.constraint(Expr.add(Expr.repeat(lamb, d, 0), beta), Domain.greaterThan(0.0))
            print("pnorm = inf. ")
        elif p_norm == 1:
            s1 = M.variable('s1', d)
            M.constraint(Expr.sub(s1, beta), Domain.greaterThan(0.0))
            M.constraint(Expr.add(s1, beta), Domain.greaterThan(0.0))
            M.constraint(Expr.sub(lamb, Expr.sum(s1)), Domain.greaterThan(0.0))
            print("pnorm = 1 ")
        else:
            M.constraint(
                Expr.hstack(
                    Var.vrepeat(lamb, n), pp
                ),
                Domain.inQCone()
            )

            M.constraint(
                Expr.hstack(
                    Var.vrepeat(lamb, n), pn
                ),
                Domain.inQCone()
            )

            print("pnorm > 1 ")

        # signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))

        # (s[n] - y*b) - (\sum{zp}*L + mulDiag(X, pp'))
        M.constraint(
            Expr.sub(
                Expr.sub(
                    s,
                    Expr.mul(y, Var.repeat(b, n))
                ),
                Expr.add(
                    Expr.mul(Expr.sum(zp, 1), L),
                    # Expr.mul(Expr.sum(zp, 1), Expr.repeat(L, n, 0)),
                    Expr.mulDiag(X, pp.transpose())
                )
            ),
            Domain.greaterThan(1.))

        # (s[n] + lamb*gamma[n] + y*b) - ((L*sum(zn)[n] + mulDiag(X[n,d], pn[n,d]'))
        M.constraint(
            Expr.sub(
                Expr.add(
                    Expr.add(
                        s,
                        Expr.repeat(Expr.mul(lamb, gamma), n, 0)
                    ),
                    Expr.mul(y, Var.repeat(b, n))
                ),
                Expr.add(
                    Expr.mul(Expr.sum(zn, 1), L),
                    # Expr.mul(Expr.sum(zn, 1), Expr.repeat(L, n, 0)),
                    Expr.mulDiag(X, pn.transpose())
                )
            ),
            Domain.greaterThan(1.))

        # zp[n,d] + (pp[n,d] + y[n*n]*beta[n*d]) >= 0
        M.constraint(
            Expr.add(
                zp,
                Expr.add(
                    pp,
                    Expr.mul(y, Expr.repeat(beta.transpose(), n, 0))
                )
            ),
            Domain.greaterThan(0.))

        # zn[n,d] + (pn[n,d] - y[n*n]*beta[n*d]) >= 0
        M.constraint(
            Expr.add(
                zn,
                Expr.sub(
                    pn,
                    Expr.mul(y, Expr.repeat(beta.transpose(), n, 0))
                )
            ),
            Domain.greaterThan(0.))
        # t >= 1/2 \sum beta_i^2
        M.constraint(Expr.vstack(self.set_reg_coef, t, beta), Domain.inRotatedQCone())
        # s[n]
        M.constraint(s, Domain.greaterThan(0.))

        return M

    def wdrosvm_space(self,X, y):
        n, d = int(X.shape[0]), int(X.shape[1])  # num samples, dimension
        M = Model()
        M.setSolverParam("optimizer", "conic")
        M.setLogHandler(open('result/WDROSVM_space.txt', 'wt'))

        beta = M.variable('beta', d)
        lamb = M.variable('Lambda')
        s = M.variable('s_i', n, Domain.greaterThan(0.))
        # t = M.variable('t', 1, Domain.equalsTo(0.0))
        t = M.variable('t', 1, Domain.unbounded())
        b = M.variable('b')

        theta = M.parameter('WasRadius')
        gamma = M.parameter('gamma')
        gamma.setValue(7)

        p_norm = self.set_dualnorm
        n_inv = 1 / self.N
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                t,
                Expr.add(
                    Expr.mul(Expr.sum(Expr.mulElm(s,self.weight)), n_inv),
                    Expr.mul(lamb, theta)
                )
            )
        )

        if p_norm == math.inf:
            M.constraint(Expr.sub(Expr.repeat(lamb, d, 0), beta), Domain.greaterThan(0.0))
            M.constraint(Expr.add(Expr.repeat(lamb, d, 0), beta), Domain.greaterThan(0.0))
            print("pnorm = inf. ")
        elif p_norm == 1:
            s1 = M.variable('s1', d)
            M.constraint(Expr.sub(s1, beta), Domain.greaterThan(0.0))
            M.constraint(Expr.add(s1, beta), Domain.greaterThan(0.0))
            M.constraint(Expr.sub(lamb, Expr.sum(s1)), Domain.greaterThan(0.0))
            print("pnorm = 1 ")
        else:
            self.pnorm(M, lamb, beta, p_norm)
            print("pnorm > 1 ")

        signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        # s[n] - (1[n] - (X[n,d]*beta[d]-b) * signs) >= 0
        # s + (X*beta-b)*signs >= 1
        M.constraint(
            Expr.add(
                s,
                Expr.mulElm(
                    Expr.sub(Expr.mul(X, beta), Var.repeat(b, n)),
                    signs
                )
            ),
            Domain.greaterThan(1.))
        # (s[n] + lamb*gamma[n]) - (1[n] + X[n,d]*beta[d]*signs)
        M.constraint(
            Expr.sub(
                Expr.add(
                    s,
                    Expr.repeat(Expr.mul(lamb, gamma), n, 0)
                ),
                Expr.mulElm(
                    Expr.sub(Expr.mul(X, beta), Var.repeat(b, n)),
                    signs
                )
            ),
            Domain.greaterThan(1.))
        # t >= 1/2 \sum beta_i^2
        M.constraint(Expr.vstack(self.set_reg_coef, t, beta), Domain.inRotatedQCone())
        return M

    def primal_svm(self,X, y):
        n, d = int(X.shape[0]), int(X.shape[1])  # num samples, dimension
        M = Model()
        M.setSolverParam("optimizer", "conic")
        M.setLogHandler(open('result/SVM.txt', 'wt'))
        beta = M.variable('beta', d, Domain.unbounded())
        # t = M.variable('t', 1, Domain.equalsTo(0.0))
        t = M.variable('t', 1, Domain.unbounded())
        b = M.variable('b', 1, Domain.unbounded())
        s = M.variable('s_i', n, Domain.greaterThan(0.))
        theta = M.parameter('WasRadius')
        n_inv = 1 / self.N
        # Expr.mulElm(s,self.weight)
        M.objective(
            ObjectiveSense.Minimize,
            Expr.add(
                t,
                Expr.mul(Expr.sum(Expr.mulElm(s,self.weight)), n_inv)
            )
        )

        signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        M.constraint(
            Expr.add(
                Expr.mulElm(signs,
                            Expr.sub(Expr.mul(X, beta), Var.repeat(b, n))
                            ),
                s
            ),
            Domain.greaterThan(1.))
        # t >= 1/2 \sum beta_i^2
        M.constraint(Expr.vstack(self.set_reg_coef, t, beta), Domain.inRotatedQCone())
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


class SimSet1(DistributionallyRobustSVM):

    def __init__(self,d, N, eps_range,X,y,X_ts,y_ts,mod_sel,weight, set_reg_coef):
        self.theta_range = eps_range
        # Fusion model instance
        super().__init__(d, N, X, y, X_ts, y_ts, mod_sel, weight, set_reg_coef)

    def simulate(self):

        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = zip(
            *[(_w, _p, _r, train_loss, train_predict, test_loss, test_predict)
              for _w, _p, _r, train_loss, train_predict, test_loss, test_predict in
              self.iter_radius(self.theta_range)])
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict

    def solve(self, epsilon):
        # Set WasRadius parameter (TrainData is already set)
        self.theta.setValue(epsilon)
        # Solve the Fusion model
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        # Portfolio weights
        out_perf=0
        beta_sol = self.beta.level()
        b_val=self.b.level()
        train_loss = self.hinge_loss(self.data_x, self.data_y, beta_sol,b_val)
        train_predict = self.F_predict(self.data_x, self.data_y, beta_sol,b_val)
        test_loss = self.hinge_loss(self.X_ts, self.y_ts, beta_sol,b_val)
        test_predict = self.F_predict(self.X_ts, self.y_ts, beta_sol,b_val)
        return beta_sol, out_perf,self.M.primalObjValue(), train_loss, train_predict, test_loss, test_predict


    def run_sim(self):
        '''
        Method to iterate over several datasets and record the results.
        '''
        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = self.iter_data()
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict


    def F_predict(self,X, y, beta,b):
        signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        z = (np.dot(X, beta) - b) * signs
        right = np.count_nonzero(z > 0) / y.shape[0]
        return right

    def hinge_loss(self,X, y, beta, b):
        signs = list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        z = (np.dot(X, beta) - b) * signs
        z_prime = 1 - z
        z_prime[z_prime < 0] = 0

        loss = np.sum(z_prime) / X.shape[0]
        return np.float(loss)