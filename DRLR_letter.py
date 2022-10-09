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
from Load_data import data_real,data_real_mnist,load_letter_mm,load_letter_alfa
from DRLR_class import SimSet1,DistributionallyRobustLG,coreset_lr,Logistic_regression
import time

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)


def Append(Train_loss1, Train_predict1, Test_loss1, Test_predict1, Time1,train_loss, train_predict,test_loss, test_predict,time_now ):
    Train_loss1.append(train_loss)
    Train_predict1.append(train_predict)
    Test_loss1.append(test_loss)
    Test_predict1.append(test_predict)
    Time1.append(time_now)
    return Train_loss1, Train_predict1, Test_loss1, Test_predict1,Time1

def Append_res(Test_loss_mean,Test_loss_var,Test_acc_mean,Test_acc_var,Time_mean,Time_var,
                Test_loss1,  Test_loss2,Test_loss3, Test_loss5,  Test_predict1,Test_predict2,Test_predict3,Test_predict5,    time1,time2,time3,time5    ):
    Test_loss_mean.append([np.mean(Test_loss1, 0),np.mean(Test_loss2, 0),np.mean(Test_loss3, 0),np.mean(Test_loss5, 0)])
    Test_loss_var.append([np.std(Test_loss1, 0),np.std(Test_loss2, 0),np.std(Test_loss3, 0),np.std(Test_loss5, 0)])
    Test_acc_mean.append([np.mean(Test_predict1, 0),np.mean(Test_predict2, 0),np.mean(Test_predict3, 0),np.mean(Test_predict5, 0)])
    Test_acc_var.append(
        [np.std(Test_predict1, 0), np.std(Test_predict2, 0), np.std(Test_predict3, 0), np.std(Test_predict5, 0)])
    Time_mean.append([np.mean(time1, 0),np.mean(time2, 0),np.mean(time3, 0),np.mean(time5, 0)])
    Time_var.append([np.std(time1, 0),np.std(time2, 0),np.std(time3, 0),np.std(time5, 0)])
    return Test_loss_mean,Test_loss_var,Test_acc_mean,Test_acc_var,Time_mean,Time_var


if __name__=="__main__":

    Epsilon_range=[[0.3]]
    Sigma = [1]
    YSigma = [0.1]

    Sample_size_index = [0.01 * i for i in range(1, 11)]

    for epsilon_range in Epsilon_range:
        class1=[load_letter_mm,load_letter_alfa]
        class2=['letter_mm','letter_alfa']
        # epsilon_range = Epsilon_range[0]
        for c_index in range(2):
            cc1=class1[c_index]
            cc2 = class2[c_index]
            path = 'result/attack/'
            if not os.path.exists(path):
                os.makedirs(path)
            path_res = 'coreset_{}_radius_{}_xpurturb_xsigma_{}_ypurturb_y_sigma{}.mat'.format(
                cc2,epsilon_range,
                Sigma[0], YSigma[0])
            Train_loss_mean, Train_loss_var, Train_acc_mean, Train_acc_var = [[] for i in range(4)]
            Test_loss_mean, Test_loss_var, Test_acc_mean, Test_acc_var, Time_mean, Time_var = [[] for i in range(6)]
            print('################################', cc2,
                  '#####################################')
            X_train, y_train, X_test, y_test = cc1()
            d = X_train.shape[1]

            Wts1, Train_loss1, Train_predict1, Test_loss1, Test_predict1, Wts2, Train_loss2, Train_predict2, Test_loss2, Test_predict2=[[] for i in range(10)]
            time1, time2 = [[] for i in range(2)]
            for t in range(1):

                N = X_train.shape[0]

                sim1 = SimSet1(X_train.shape[1], N, epsilon_range, X_train, y_train, X_test, y_test, 0,
                               np.ones((X_train.shape[0], 1)))
                #
                # # 200 simulations...
                wts, perf, rel, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                Train_loss1, Train_predict1, Test_loss1, Test_predict1, time1 = Append(Train_loss1,
                                                                                       Train_predict1,
                                                                                       Test_loss1,
                                                                                       Test_predict1, time1,
                                                                                       train_loss,
                                                                                       train_predict,
                                                                                       test_loss, test_predict,
                                                                                       sim1.sol_time[0])
                print('1')
                print("Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                    X_train.shape[0], sim1.sol_time[0]), ' acc: ', test_predict)
                sim1.M.dispose()
                del (sim1)
                gc.collect()
                sim1 = SimSet1(X_train.shape[1], N, epsilon_range, X_train, y_train, X_test, y_test, 1,
                               np.ones((X_train.shape[0], 1)))
                #
                # # 200 simulations...
                wts2, perf2, rel2, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                Train_loss2, Train_predict2, Test_loss2, Test_predict2, time2 = Append(Train_loss2,
                                                                                       Train_predict2,
                                                                                       Test_loss2,
                                                                                       Test_predict2, time2,
                                                                                       train_loss,
                                                                                       train_predict,
                                                                                       test_loss, test_predict,
                                                                                       sim1.sol_time[0])
                print('2')
                print("Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                    X_train.shape[0], sim1.sol_time[0]), ' acc: ', test_predict)
                sim1.M.dispose()
                del (sim1)
                gc.collect()

            for sample_size_index in Sample_size_index:
                time3, time4, time5 = [[] for i in range(3)]
                Wts3, Train_loss3, Train_predict3, Test_loss3, Test_predict3,  Wts5, Train_loss5, Train_predict5, Test_loss5, Test_predict5 = [
                    [] for i in range(10)]
                for t in range(50):
                    N = X_train.shape[0]


                    Sample_size = int(sample_size_index * X_train.shape[0])

                    index_new = np.random.permutation(X_train.shape[0])[0:Sample_size]
                    X_train_sample = X_train[index_new, :]
                    y_train_sample = y_train[index_new, :]



                    print('3 ',X_train_sample.shape[0])
                    sim1 = SimSet1(X_train_sample.shape[1], N, epsilon_range, X_train_sample, y_train_sample,
                                   X_test,
                                   y_test, 0, np.ones((X_train_sample.shape[0], 1)) * N / Sample_size)
                    wts3, perf3, rel3, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                    Train_loss3, Train_predict3, Test_loss3, Test_predict3, time3 = Append(Train_loss3,
                                                                                           Train_predict3,
                                                                                           Test_loss3,
                                                                                           Test_predict3, time3,
                                                                                           train_loss,
                                                                                           train_predict,
                                                                                           test_loss, test_predict,
                                                                                           sim1.sol_time[0])

                    print("Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                        X_train_sample.shape[0], sim1.sol_time[0]), ' acc: ', test_predict)

                    b_val = sim1.b.level()
                    T = sim1.sol_time[0]
                    coreset_size = Sample_size
                    lamda, gamma = sim1.lamb.level(), sim1.gamma
                    sim1.M.dispose()
                    del (sim1)
                    gc.collect()


                    start = time.time()
                    X_coreset, y_coreset, weight = coreset_lr(X_train, y_train, wts3[0], b_val, coreset_size, lamda,
                                                              gamma,
                                                              Logistic_regression)
                    T += time.time() - start
                    print('5 ', X_coreset.shape[0])
                    sim1 = SimSet1(X_coreset.shape[1], N, epsilon_range, X_coreset, y_coreset, X_test,
                                   y_test,  0, weight)
                    wts5, perf5, rel5, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                    T += sim1.sol_time[0]
                    Train_loss5, Train_predict5, Test_loss5, Test_predict5, time5 = Append(Train_loss5,
                                                                                           Train_predict5,
                                                                                           Test_loss5,
                                                                                           Test_predict5, time5,
                                                                                           train_loss,
                                                                                           train_predict,
                                                                                           test_loss, test_predict,
                                                                                           T)
                    print("Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                        X_coreset.shape[0], sim1.sol_time[0]), ' acc: ', test_predict)

                    sim1.M.dispose()
                    del (sim1)
                    gc.collect()

                Test_loss_mean, Test_loss_var, Test_acc_mean, Test_acc_var, Time_mean, Time_var = Append_res(
                    Test_loss_mean, Test_loss_var, Test_acc_mean, Test_acc_var, Time_mean, Time_var,
                    Test_loss1, Test_loss2, Test_loss3, Test_loss5, Test_predict1, Test_predict2, Test_predict3,
                    Test_predict5, time1, time2, time3, time5)
            print('Acc ',Test_acc_mean)
            print('Loss',Test_loss_mean)
            scio.savemat(
                path +path_res,
                {'Test_loss_mean': np.array(Test_loss_mean), 'Test_loss_var': np.array(Test_loss_var), 'Test_acc_mean': np.array(Test_acc_mean),
                 'Test_acc_var': np.array(Test_acc_var), 'Time_mean': np.array(Time_mean),'Time_var':np.array(Time_var),
                 })

