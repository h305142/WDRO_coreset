from Load_data import *
from SVM_class import SimSet1
from coreset import svm_query_h, svm_coreset2
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
    Epsilon_range = [[0.2]]
    Sigma = [1]
    YSigma = [0.1]
    set_reg_coef = 200
    Sample_size_index = [0.01 * i for i in range(1, 11)]

    class1=[i for i in range(10)]
    class2=[i for i in range(10)]
    for epsilon_range in Epsilon_range:
        for sample_size_index in Sample_size_index:
            path = 'result/letter/'
            if not os.path.exists(path):
                os.makedirs(path)
            path_res = 'letter_mm_r_{}_xsigma_{}_y_sigma{}_samp_{}.mat'.format(
                epsilon_range,
                Sigma[0], YSigma[0],sample_size_index)
            Test_loss_mean, Test_loss_var, Test_acc_mean, Test_acc_var, Time_mean, Time_var = [[] for i in range(6)]
            for c1 in range(1):
                for c2 in range(1):
                    cc1=class1[c1]
                    cc2 = class2[c2]
                    print('################################', cc1, cc2, '#####################################')
                    X_tr, y_tr, X_ts, y_ts = load_letter_mm()
                    d = X_tr.shape[1]
                    Wts1, Train_loss1, Train_predict1, Test_loss1, Test_predict1, Wts2, Train_loss2, Train_predict2, Test_loss2, Test_predict2=[[] for i in range(10)]
                    time1, time2 = [[] for i in range(2)]

                    N = X_tr.shape[0]
                    for t in range(1):
                        sim1 = SimSet1(X_tr.shape[1], N, epsilon_range, X_tr, y_tr, X_ts, y_ts, 1,
                                       np.ones((X_tr.shape[0], 1)), set_reg_coef)

                        wts, perf, rel, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                        Train_loss1, Train_predict1, Test_loss1, Test_predict1, time1 = Append(Train_loss1,
                                                                                               Train_predict1,
                                                                                               Test_loss1,
                                                                                               Test_predict1, time1,
                                                                                               train_loss,
                                                                                               train_predict,
                                                                                               test_loss, test_predict,
                                                                                               sim1.sol_time[0])

                        print("1 Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                            N, sim1.sol_time[0]), ' acc: ', test_predict)
                        sim1.M.dispose()
                        del (sim1)
                        gc.collect()

                        sim1 = SimSet1(X_tr.shape[1], N, epsilon_range, X_tr, y_tr, X_ts, y_ts, 0,
                                       np.ones((X_tr.shape[0], 1)), set_reg_coef)

                        wts2, perf2, rel2, train_loss, train_predict, test_loss, test_predict = sim1.run_sim()
                        Train_loss2, Train_predict2, Test_loss2, Test_predict2, time2 = Append(Train_loss2,
                                                                                               Train_predict2,
                                                                                               Test_loss2,
                                                                                               Test_predict2, time2,
                                                                                               train_loss,
                                                                                               train_predict,
                                                                                               test_loss, test_predict,
                                                                                               sim1.sol_time[0])

                        print("2 Time taken in initial solve of model with N={0}: {1:.4f} s".format(
                            N, sim1.sol_time[0]), ' acc: ', test_predict)
                        sim1.M.dispose()
                        del (sim1)
                        gc.collect()


                    time3, time4, time5 = [[] for i in range(3)]
                    Wts3, Train_loss3, Train_predict3, Test_loss3, Test_predict3,  Wts5, Train_loss5, Train_predict5, Test_loss5, Test_predict5 = [
                        [] for i in range(10)]
                    for t in range(50):
                        Sample_size = int(sample_size_index * X_tr.shape[0])

                        index_new = np.random.permutation(X_tr.shape[0])[0:Sample_size]
                        X_tr_sample = X_tr[index_new, :]
                        y_tr_sample = y_tr[index_new, :]


                        print('3 ',X_tr_sample.shape[0])
                        sim1 = SimSet1(X_tr_sample.shape[1], N, epsilon_range, X_tr_sample, y_tr_sample,
                                       X_ts,
                                       y_ts, 1, np.ones((X_tr_sample.shape[0], 1)) * N / Sample_size, set_reg_coef)
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
                            N, sim1.sol_time[0]), ' acc: ', test_predict)

                        b_val = sim1.b.level()
                        T = sim1.sol_time[0]
                        coreset_size = Sample_size
                        lamda, gamma = sim1.lamb.level(), sim1.gamma
                        sim1.M.dispose()
                        del (sim1)
                        gc.collect()

                        start = time.time()
                        X_coreset, y_coreset, weight = svm_coreset2(X_tr, y_tr, coreset_size, wts3[0], b_val, lamda,
                                                                  gamma,
                                                                  svm_query_h)
                        T += time.time() - start
                        print('5 ', X_coreset.shape[0])
                        sim1 = SimSet1(X_coreset.shape[1], N, epsilon_range, X_coreset, y_coreset, X_ts,
                                       y_ts, 1, weight, set_reg_coef)
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
                        print("Time taken in initial solve of model with N={0:.4f}: {1:.4f} s".format(
                            T, sim1.sol_time[0]), ' acc: ', test_predict)

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
