import numpy as np
import time

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

def huber_query_h(X, y, beta, b_val, delta):
    y_pred = np.dot(X, beta) + b_val
    z = np.absolute(y_pred - y.squeeze())
    huber_loss_vec = np.zeros(z.shape)
    idx_l = np.where(z > delta)[0]
    idx_s = np.where(z <= delta)[0]
    # print('num > delta: ', idx_l.shape)
    huber_loss_vec[idx_l] = delta * (z[idx_l] - 0.5 * delta)
    huber_loss_vec[idx_s] = (z[idx_s] ** 2) * 0.5
    loss_h = np.sum(huber_loss_vec)
    return huber_loss_vec



def Cal_sample_base(coreset_size,N,k=1):
    All_size=0
    for i in range(N+1):
        All_size+=np.power(1+1/(np.power(2,i)*k),2)
    return np.floor(coreset_size/All_size)


def svm_coreset2(X, y, coreset_size,beta, b,lamb, gamma, function):
    n, d = X.shape
    f = function(X, y, beta, b, lamb, gamma)
    X = np.hstack((X, y))
    L_max = np.max(f, 0)
    H = np.sum(f, 0) / X.shape[0]
    N = int(np.ceil(np.log2(L_max / H)))
    # print(L_max, H, N)
    kk = 1
    sample_base = Cal_sample_base(coreset_size=coreset_size, N=N, k=kk)
    coreset = []
    Weight = []
    for i in range(1, N + 1):
        index_i = np.array(np.where((f > H * pow(2, i - 1)) & (f <= H * pow(2, i))))[0]
        sample_num_i = int(sample_base * np.power(1 + 1 / (np.power(2, i) * kk), 2))
        if sample_num_i > 0:
            if len(coreset) == 0:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = X[choice, :]
                    Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
                else:
                    coreset = X[index_i, :]
                    Weight = np.ones((len(index_i), 1))
            else:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = np.vstack((coreset, X[choice, :]))
                    Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
                else:
                    coreset = np.vstack((coreset, X[index_i, :]))
                    Weight = np.vstack((Weight, np.ones((len(index_i), 1))))

            print('layer{}:{}'.format(i,sample_num_i))
    index_i = np.array(np.where(f <= H))[0]
    sample_num_i = coreset_size - len(coreset)
    if len(coreset) == 0:
        if sample_num_i <= index_i.shape[0]:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            coreset = X[choice, :]
            Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
        else:
            coreset = X[index_i, :]
            Weight = np.ones((len(index_i), 1))
    else:
        if sample_num_i <= index_i.shape[0]:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            coreset = np.vstack((coreset, X[choice, :]))
            Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
        else:
            coreset = np.vstack((coreset, X[index_i, :]))
            Weight = np.vstack((Weight, np.ones((len(index_i), 1))))
    print('layer0:{}'.format(sample_num_i))
    print('finalsize:', coreset.shape[0])
    return coreset[:, 0:d ], coreset[:, d:], Weight


def huber_coreset(X, y, coreset_size,beta, b, delta, function):
    n, d = X.shape
    f = function(X, y, beta, b, delta)
    X = np.hstack((X, y))
    L_max = np.max(f, 0)
    H = np.sum(f, 0) / X.shape[0]
    N = int(np.ceil(np.log2(L_max / H)))
    # print(L_max, H, N)
    kk = 1
    sample_base = Cal_sample_base(coreset_size=coreset_size, N=N, k=kk)
    coreset = []
    Weight = []
    for i in range(1, N + 1):
        index_i = np.array(np.where((f > H * pow(2, i - 1)) & (f <= H * pow(2, i))))[0]
        sample_num_i = int(sample_base * np.power(1 + 1 / (np.power(2, i) * kk), 2))
        if sample_num_i > 0:
            if len(coreset) == 0:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = X[choice, :]
                    Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
                else:
                    coreset = X[index_i, :]
                    Weight = np.ones((len(index_i), 1))
            else:
                if sample_num_i <= index_i.shape[0]:
                    choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
                    coreset = np.vstack((coreset, X[choice, :]))
                    Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
                else:
                    coreset = np.vstack((coreset, X[index_i, :]))
                    Weight = np.vstack((Weight, np.ones((len(index_i), 1))))

            print('layer{}:{}'.format(i,sample_num_i))
    index_i = np.array(np.where(f <= H))[0]
    sample_num_i = coreset_size - len(coreset)
    if len(coreset) == 0:
        if sample_num_i <= index_i.shape[0]:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            coreset = X[choice, :]
            Weight = np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)
        else:
            coreset = X[index_i, :]
            Weight = np.ones((len(index_i), 1))
    else:
        if sample_num_i <= index_i.shape[0]:
            choice = index_i[np.random.permutation(index_i.shape[0])[0:sample_num_i]]
            coreset = np.vstack((coreset, X[choice, :]))
            Weight = np.vstack((Weight, np.ones((sample_num_i, 1)) * (index_i.shape[0] / sample_num_i)))
        else:
            coreset = np.vstack((coreset, X[index_i, :]))
            Weight = np.vstack((Weight, np.ones((len(index_i), 1))))
    print('layer0:{}'.format(sample_num_i))
    print('finalsize:', coreset.shape[0])
    return coreset[:, 0:d ], coreset[:, d:], Weight
