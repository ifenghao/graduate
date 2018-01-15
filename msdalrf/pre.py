# coding:utf-8

import numpy as np
from scipy.misc import imresize
from util_lrf import deploy
from multiprocessing import Pool

epsilon = 1e-3
dataset_path = '/home/zfh/dataset/'


def make_dataset():
    tr_pos = None
    for i in range(3):
        one = np.load(dataset_path + 'tr_pos' + str(i) + '.npy')
        tr_pos = np.concatenate((tr_pos, one), axis=0) if tr_pos is not None else one
    tr_pos_label = np.ones(tr_pos.shape[0])
    te_pos = None
    for i in range(3):
        one = np.load(dataset_path + 'te_pos' + str(i) + '.npy')
        te_pos = np.concatenate((te_pos, one), axis=0) if te_pos is not None else one
    te_pos_label = np.ones(te_pos.shape[0])
    tr_neg = None
    for i in range(3):
        one = np.load(dataset_path + 'tr_neg' + str(i) + '.npy')
        tr_neg = np.concatenate((tr_neg, one), axis=0) if tr_neg is not None else one
    tr_neg_label = np.zeros(tr_neg.shape[0])
    te_neg = None
    for i in range(3):
        one = np.load(dataset_path + 'te_neg' + str(i) + '.npy')
        te_neg = np.concatenate((te_neg, one), axis=0) if te_neg is not None else one
    te_neg_label = np.zeros(te_neg.shape[0])

    tr_X = np.concatenate((tr_pos, tr_neg), axis=0)
    tr_y = np.concatenate((tr_pos_label, tr_neg_label), axis=0)
    rand_idx = np.random.permutation(tr_X.shape[0])
    tr_X = tr_X[rand_idx]
    tr_y = tr_y[rand_idx]

    te_X = np.concatenate((te_pos, te_neg), axis=0)
    te_y = np.concatenate((te_pos_label, te_neg_label), axis=0)
    rand_idx = np.random.permutation(te_X.shape[0])
    te_X = te_X[rand_idx]
    te_y = te_y[rand_idx]

    np.save(dataset_path + 'tr_X.npy', tr_X)
    np.save(dataset_path + 'tr_y.npy', tr_y)
    np.save(dataset_path + 'te_X.npy', te_X)
    np.save(dataset_path + 'te_y.npy', te_y)


def dataset_resize():
    for fname in ('tr_X', 'te_X'):
        X = np.load(dataset_path + fname + '.npy')
        X = X.transpose((0, 2, 3, 1))
        new_X = None
        for i in xrange(X.shape[0]):
            tmp = imresize(X[i], (224, 224))
            tmp = tmp.transpose((2, 0, 1))[None, :, :, :]
            new_X = np.concatenate((new_X, tmp), axis=0) if new_X is not None else tmp
        np.save(dataset_path + fname + '_s.npy', new_X)


def decomp(batch, n_jobs):
    starts = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    starts[:batch % n_jobs] += 1
    return starts


def job(X, shape):
    new_X = None
    for i in xrange(X.shape[0]):
        tmp = imresize(X[i], shape)
        tmp = tmp.transpose((2, 0, 1))[None, :, :, :]
        new_X = np.concatenate((new_X, tmp), axis=0) if new_X is not None else tmp
    return new_X


def dataset_resize_mcpu(splits=12):
    for fname in ('tr_X', 'te_X'):
        X = np.load(dataset_path + fname + '.npy')
        X = X.transpose((0, 2, 3, 1))
        starts = decomp(X.shape[0], splits)
        pool = Pool(processes=splits)
        jobs = []
        for i in xrange(splits):
            tmp = X[:starts[i]]
            X = X[starts[i]:]
            jobs.append(pool.apply_async(job, (tmp, (64, 64))))
        pool.close()
        pool.join()
        new_X = None
        for one_job in jobs:
            tmp = one_job.get()
            new_X = np.concatenate([new_X, tmp], axis=0) if new_X is not None else tmp
        np.save(dataset_path + fname + '_t.npy', new_X)


def dataset_resize_split(split=3):
    for fname in ('tr_X', 'te_X'):
        X = np.load(dataset_path + fname + '.npy')
        X = X.transpose((0, 2, 3, 1))
        starts = deploy(X.shape[0], split)
        new_X = None
        cnt = 0
        for i in xrange(X.shape[0]):
            tmp = imresize(X[i], (224, 224))
            tmp = tmp.transpose((2, 0, 1))[None, :, :, :]
            new_X = np.concatenate((new_X, tmp), axis=0) if new_X is not None else tmp
            if i in starts:
                np.save(dataset_path + fname + '_t' + str(cnt) + '.npy', new_X)
                cnt += 1
                new_X = None


def load_train(size='_t'):
    Xname = 'tr_X' + size + '.npy'
    tr_X = np.load(dataset_path + Xname)
    tr_y = np.load(dataset_path + 'tr_y.npy')
    return tr_X, tr_y


def load_unlabel():
    X = np.load(dataset_path + 'unlabel_t.npy')
    return X


def load_test(size='_t'):
    Xname = 'te_X' + size + '.npy'
    te_X = np.load(dataset_path + Xname)
    te_y = np.load(dataset_path + 'te_y.npy')
    return te_X, te_y


def one_hot(x, n):
    x = np.array(x, dtype=np.int)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def norm2d(tr_X, vate_X):
    avg = np.mean(tr_X, axis=None, dtype=np.float32, keepdims=True)
    # var = np.var(tr_X, axis=None, dtype=theano.config.floatX, keepdims=True)
    return (tr_X - avg) / 127.5, (vate_X - avg) / 127.5


# def norm4d(tr_X, vate_X):
#     avg = np.mean(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
#     # var = np.var(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
#     return (tr_X - avg) / 127.5, (vate_X - avg) / 127.5

def norm4d(X, avg=None, var=None):
    if avg is None and var is None:
        avg = np.mean(X, axis=(0, 2, 3), dtype=np.float32, keepdims=True)
        var = np.var(X, axis=(0, 2, 3), dtype=np.float32, keepdims=True)
        return (X - avg) / (np.sqrt(var + epsilon)), avg, var
    else:
        return (X - avg) / (np.sqrt(var + epsilon))


# pylearn2
def norm4d_per_sample(X, scale=1., reg=0.1, cross_ch=False):
    Xshape = X.shape
    X = X.reshape((Xshape[0] * Xshape[1], -1)) if cross_ch \
        else X.reshape((Xshape[0], -1))
    mean = X.mean(axis=1)
    X = X - mean[:, None]
    normalizer = np.sqrt((X ** 2).mean(axis=1) + reg) / scale
    X = X / normalizer[:, None]
    return X.reshape(Xshape)


if __name__ == '__main__':
    # make_dataset()
    dataset_resize_mcpu()
