

import numpy as np

from mosek.fusion import *

def Logistic_regression(X, y, beta,b,lamd,gamma):
    xbeta =-y*( np.dot(X, beta[:,np.newaxis]) + b)
    f1 =np.log( 1+np.exp( xbeta) )
    f2=np.log( 1+np.exp( -1*xbeta) )-lamd*gamma
    F=np.hstack((f1,f2))
    return np.max(F,axis=1)

def Cal_sample_base(coreset_size,N,k=1):
    All_size=0
    for i in range(N+1):
        All_size+=np.power(1+1/(np.power(2,i)*k),2)
    return np.floor(coreset_size/All_size)

def coreset_lr(X, y,  beta, b, coreset_size,lamb, gamma, function):
    n, d = X.shape
    f = function(X, y, beta, b, lamb, gamma)
    offset = np.min(f, 0)
    f = f - offset
    X = np.hstack((X,y))
    L_max = np.max(f, 0)
    H = np.sum(f) / n
    H = H

    N = int(np.ceil(np.log2(L_max / H)))
    coreset = np.array([])
    Weight = np.array([])
    index_i = np.array(np.where(f <= H))[0]

    sample_num_i = int(coreset_size * index_i.shape[0] / X.shape[0])

    if sample_num_i > 0:
        if sample_num_i <= index_i.shape[0]:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            coreset = X[choice, :]
            Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
        else:
            coreset = X[index_i, :]
            Weight = np.ones((len(index_i), 1))
    for i in range(1, N + 1):
        index_i = np.array(np.where((f > H * pow(2, i - 1)) & (f <= H * pow(2, i))))[0]
        # sample_num_i = int(sample_base * np.power(1 + 1 / (np.power(2, i) * k), 2))
        sample_num_i = int(coreset_size * index_i.shape[0] / X.shape[0])
        if sample_num_i > 0:
            if sample_num_i <= index_i.shape[0]:
                choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                coreset = np.vstack((coreset, X[choice, :]))
                Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
            else:
                coreset = np.vstack((coreset, X[index_i, :]))
                Weight = np.vstack((Weight, np.ones((len(index_i), 1))))
    sample_num_i = coreset_size - len(coreset)
    if sample_num_i <= 0:
        Weight = Weight.squeeze()
        shuffle_idx = np.random.permutation(coreset.shape[0])
        coreset = coreset[shuffle_idx, :]
        Weight = Weight[shuffle_idx]
        return coreset[:, :d], coreset[:, d:], Weight
    ind = np.random.permutation(n)[0:sample_num_i]
    C_u = X[ind, :]

    Weight = Weight * (n - sample_num_i) / n
    coreset = np.vstack((coreset, C_u))
    Weight = np.vstack((Weight, np.ones((sample_num_i, 1))))
    shuffle_idx = np.random.permutation(coreset.shape[0])
    coreset = coreset[shuffle_idx, :]
    Weight = Weight[shuffle_idx]
    return coreset[:, :d], coreset[:, d:], Weight



class DistributionallyRobustLG:

    def __init__(self, m, N,X,y,X_test,y_test,LG_select,weight):
        self.m, self.N = m, N
        self.gamma=7
        self.weight=weight
        if LG_select==0:
             self.M = self.logisticRegression(m, N,X,y,0)
        else:
            self.M = self.logisticRegression2(m, N, X, y, 2)

        self.beta = self.M.getVariable('beta')
        self.b=self.M.getVariable('b')
        self.lamb=self.M.getVariable('Lambda')
        self.sol_time = []
        self.data_x=X
        self.data_y=y
        self.X_test=X_test
        self.y_test=y_test


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

    def logisticRegression(self,d, n,X,y,p):
        M = Model()
        M.setSolverParam("optimizer", "conic")
        beta = M.variable('beta',d)
        s = M.variable('s_i', X.shape[0])
        lamb = M.variable('Lambda')
        theta = M.parameter('WasRadius')
        b=M.variable('b',1)
        e1=Expr.repeat(b,X.shape[0],0)
        kappa = self.gamma
        n_inv=1/n
        M.objective(ObjectiveSense.Minimize,
                    Expr.add(Expr.mul(Expr.sum(Expr.mulElm(s, self.weight)), n_inv), Expr.mul(lamb, theta)))
        M.constraint(Expr.sub(beta,Expr.repeat(lamb,d,0)), Domain.lessThan(0.0))
        M.constraint(Expr.add(beta,Expr.repeat(lamb,d,0)), Domain.greaterThan(0.0))
        signs = list(map(lambda y: -1.0 if y == 1 else 1.0, y))
        sign2=list(map(lambda y: 1.0 if y == 1 else -1.0, y))
        self.softplus(M, s , Expr.mulElm(Expr.add(Expr.mul(X, beta),e1), signs))
        self.softplus(M, Expr.add(s,Expr.repeat(Expr.mul(lamb,kappa), X.shape[0],0)) , Expr.mulElm(Expr.add(Expr.mul(X, beta),e1), sign2))
        return M

    def logisticRegression2(self,d, n,X,y,p):
        M = Model()
        M.setSolverParam("optimizer", "conic")
        s = M.variable('s_i',X.shape[0])
        theta = M.parameter('WasRadius')
        beta = M.variable('beta',d)
        b_val = M.variable('b', 1)
        e1 = Expr.repeat(b_val, X.shape[0], 0)
        reg = M.variable()
        lamb=0.1

        n_inv = 1 / n
        M.objective(ObjectiveSense.Minimize, Expr.add( Expr.mul(Expr.sum(s),n_inv),Expr.mul(lamb,reg)))
        M.constraint(Var.vstack(reg, beta), Domain.inQCone())

        signs = list(map(lambda y: -1.0 if y == 1 else 1.0, y))
        self.softplus(M, s, Expr.mulElm(Expr.add(Expr.mul(X, beta),e1), signs))
        return M

    def sample_average(self, x, t, data):
        '''
        Calculate the sample average approximation for given x and tau.
        '''
        l = np.matmul(data, x)
        return np.mean(np.maximum(-l + 10*t, -51*l - 40*t))

    def iter_data(self):
        '''
        Generator method for iterating through values for the
        TrainData parameter.
        '''
        return  self.simulate()

    def iter_radius(self, epsilon_range):
        '''
        Generator for iterating through values for the WasRadius
        parameter.
        '''
        for epsilon in epsilon_range:
            yield self.solve(epsilon)

    def simulate(self):
        '''
        Define in child classes.
        '''
        pass

    def solve(self, epsilon):
        '''
        Define in child classes.
        '''
        pass


class SimSet1(DistributionallyRobustLG):

    def __init__(self,d, N, eps_range,X,y,X_test,y_test,LG_select,weight):
        self.theta_range = eps_range
        super().__init__(d, N, X, y, X_test, y_test, LG_select, weight)

    def simulate(self):
        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = zip(
            *[(_w, _p, _r, train_loss, train_predict, test_loss, test_predict)
              for _w, _p, _r, train_loss, train_predict, test_loss, test_predict in
              self.iter_radius(self.theta_range)])
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict

    def solve(self, epsilon):
        self.theta.setValue(epsilon)
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        beta_sol = self.beta.level()
        b_val=self.b.level()
        out_perf = 0
        train_loss = self.l2_loss_test(self.data_x, self.data_y, beta_sol[:, np.newaxis],b_val)
        train_predict = self.F_predict(self.data_x, self.data_y, beta_sol[:, np.newaxis],b_val)
        test_loss = self.l2_loss_test(self.X_test, self.y_test, beta_sol[:, np.newaxis],b_val)
        test_predict = self.F_predict(self.X_test, self.y_test, beta_sol[:, np.newaxis],b_val)
        return beta_sol, out_perf, self.M.primalObjValue(), train_loss, train_predict, test_loss, test_predict


    def run_sim(self):
        beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict = self.iter_data()
        return beta_sol, perf, rel, train_loss, train_predict, test_loss, test_predict


    def F_predict(self,X, y, beta,b_val):
        y_predict = np.dot(X, beta)+np.repeat(b_val,X.shape[0])[:,np.newaxis]
        y[y == 0] = -1
        right = np.count_nonzero(y_predict * y > 0) / y.shape[0]
        return np.float(right)

    def l2_loss_test(self,X_test, y_test, beta,b_val):
        xbeta = np.dot(X_test, beta)+np.repeat(b_val,X_test.shape[0])[:,np.newaxis]
        loss=np.sum(  np.log(1 + np.exp(-y_test * xbeta)), 0) / X_test.shape[0]
        return np.float(loss)