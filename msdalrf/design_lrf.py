# coding:utf-8
import numpy as np
from scipy import stats
from multiprocessing import Pool
import time
import gc

__all__ = ['collaborate_mn', 'collaborate_mn_jobs', 'add_Q_noise', 'add_P_noise']


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def deploy2(batch, n_jobs):
    starts = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    starts[:batch % n_jobs] += 1
    return starts


def relu_probs_mn(X, W, p_noise):  # noise表示置零的概率
    n_feature, n_hidden = W.shape
    hidden_positive_prob = []
    for i in xrange(n_hidden):
        X_hidden = X * W[:, i]
        mu = np.sum(X_hidden[:, :-1], axis=1) * (1. - p_noise)
        mu += X_hidden[:, -1]  # 偏置期望为本身
        sigma = np.sqrt(np.sum(X_hidden[:, :-1] ** 2, axis=1) * (1. - p_noise) * p_noise)  # 偏置方差为0
        col_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
        hidden_positive_prob.append(col_positive_prob[:, None])
    hidden_positive_prob = np.concatenate(hidden_positive_prob, axis=1)
    return hidden_positive_prob


# 快速实现
def add_Q_noise(S_X, p_noise):
    n_feature = S_X.shape[0]
    S_X *= (1. - p_noise) ** 2
    diag_idx = np.diag_indices(n_feature - 1)
    S_X[diag_idx] /= 1. - p_noise
    S_X[-1, :] /= 1. - p_noise
    S_X[:, -1] /= 1. - p_noise
    return S_X


def design_Q_mn(X, W, P_positive, p_noise):
    n_sample, n_feature = X.shape
    n_feature, n_hidden = W.shape
    Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
    for i in xrange(n_sample):
        X_row = X[[i], :]
        S_X = np.dot(X_row.T, X_row)
        S_X = add_Q_noise(S_X, p_noise)
        P_row = P_positive[i, :]
        W_p = W * P_row
        half_p = np.dot(W_p.T, S_X)
        Q_i = np.dot(half_p, W_p)
        Q_i_diag = np.sum(half_p * W.T, axis=1)
        diag_idx = np.diag_indices(n_hidden)
        Q_i[diag_idx] = Q_i_diag
        Q += Q_i
    return Q


def add_P_noise(S_X, p_noise):
    S_X *= 1. - p_noise
    S_X[-1, :] /= 1. - p_noise
    return S_X


def design_P_mn(X, W, P_positive, p_noise):
    n_sample, n_feature = X.shape
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_sample):
        X_row = X[[i], :]
        S_X = X_row.T.dot(X_row)[:, :-1]  # 最后一列为偏置
        S_X = add_P_noise(S_X, p_noise)
        P_row = P_positive[i, :]
        W_p = W * P_row
        P += np.dot(W_p.T, S_X)
    return P


def collaborate_mn(X, W, p_noise, splits=100):
    n_sample = X.shape[0]
    starts = deploy(n_sample, splits)
    Q = 0.
    P = 0.
    for i in xrange(splits):
        start = time.time()
        P_positive = relu_probs_mn(X[starts[i]:starts[i + 1]], W, p_noise)
        Q += design_Q_mn(X[starts[i]:starts[i + 1]], W, P_positive, p_noise)
        P += design_P_mn(X[starts[i]:starts[i + 1]], W, P_positive, p_noise)
        print i, time.time() - start
    return Q, P


def job_mn(X, W, p_noise):
    start = time.time()
    P_positive = relu_probs_mn(X, W, p_noise)
    Q = design_Q_mn(X, W, P_positive, p_noise)
    P = design_P_mn(X, W, P_positive, p_noise)
    print time.time() - start
    return Q, P


def collaborate_mn_jobs(X, W, p_noise, splits=10):
    n_sample = X.shape[0]
    starts = deploy2(n_sample, splits)
    pool = Pool(processes=splits)
    jobs = []
    for i in xrange(splits):
        tmp = X[:starts[i]]
        X = X[starts[i]:]
        jobs.append(pool.apply_async(job_mn, (tmp, W, p_noise)))
        del tmp
        gc.collect()
    pool.close()
    pool.join()
    Q = 0.
    P = 0.
    for one_job in jobs:
        Qtmp, Ptmp = one_job.get()
        Q += Qtmp
        P += Ptmp
    return Q, P


########################################################################################################################


def relu_probs_gs(X, W, s_noise):  # noise表示高斯标准差
    mu = np.dot(X, W)  # 期望
    sigma = np.sqrt(np.sum(W[:-1, :] ** 2, axis=0))  # 方差,偏置方差为0
    sigma *= s_noise
    hidden_positive_prob = 1. - stats.norm.cdf(-mu / sigma)
    return hidden_positive_prob


def design_Q_gs(X, W, P_positive):  # gs和mn相比仅仅不需要add_Q_noise
    n_sample, n_feature = X.shape
    n_feature, n_hidden = W.shape
    Q = 0.  # np.zeros((n_hidden, n_hidden), dtype=float)
    for i in xrange(n_sample):
        X_row = X[[i], :]
        S_X = np.dot(X_row.T, X_row)
        P_row = P_positive[i, :]
        W_p = W * P_row
        half_p = np.dot(W_p.T, S_X)
        Q_i = np.dot(half_p, W_p)
        Q_i_diag = np.sum(half_p * W.T, axis=1)
        diag_idx = np.diag_indices(n_hidden)
        Q_i[diag_idx] = Q_i_diag
        Q += Q_i
    return Q


def design_P_gs(X, W, P_positive):  # gs和mn相比仅仅不需要add_P_noise
    n_sample, n_feature = X.shape
    P = 0.  # np.zeros((n_hidden, n_feature), dtype=float)
    for i in xrange(n_sample):
        X_row = X[[i], :]
        S_X = X_row.T.dot(X_row)[:, :-1]  # 最后一列为偏置
        P_row = P_positive[i, :]
        P += np.dot((W * P_row).T, S_X)
    return P


def collaborate_gs(X, W, s_noise, splits=10):
    n_sample = X.shape[0]
    starts = deploy(n_sample, splits)
    Q = 0.
    P = 0.
    for i in xrange(splits):
        start = time.time()
        P_positive = relu_probs_gs(X[starts[i]:starts[i + 1]], W, s_noise)
        Q += design_Q_gs(X[starts[i]:starts[i + 1]], W, P_positive)
        P += design_P_gs(X[starts[i]:starts[i + 1]], W, P_positive)
        print i, time.time() - start
    return Q, P
