import numpy as np
from numpy.linalg import solve
from copy import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import theano
import theano.tensor as T
from theano.sandbox.neighbours import images2neibs
from lasagne.theano_extensions.padding import pad as lasagnepad
from theano.tensor.signal.pool import pool_2d
from copy import deepcopy
from sklearn import svm
from multiprocessing import Pool

epsilon = 1e-3
dataset_path = '/home/zfh/dataset/'


def accuracy(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    return accuracy_score(ytrue, ypred)


def precision_recall(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    p, r, _, _ = precision_recall_fscore_support(ytrue, ypred, labels=[1])
    return p[0], r[0]


def dice(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    intersection = np.logical_and(x, y)
    print intersection
    return 2. * intersection.sum() / (x.sum() + y.sum())


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


def norm4d_per_sample(X, scale=1., reg=0.1, cross_ch=False):
    Xshape = X.shape
    X = X.reshape((Xshape[0] * Xshape[1], -1)) if cross_ch \
        else X.reshape((Xshape[0], -1))
    mean = X.mean(axis=1)
    X = X - mean[:, None]
    normalizer = np.sqrt((X ** 2).mean(axis=1) + reg) / scale
    X = X / normalizer[:, None]
    return X.reshape(Xshape)


def im2col_compfn(shape, fsize, stride, pad, ignore_border=False):
    assert len(shape) == 2
    assert isinstance(pad, int)
    if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
    if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
    X = T.tensor4()
    if not ignore_border:
        rows, cols = shape
        rows, cols = rows + 2 * pad, cols + 2 * pad
        rowpad = colpad = 0
        rowrem = (rows - fsize[0]) % stride[0]
        if rowrem: rowpad = stride[0] - rowrem
        colrem = (cols - fsize[1]) % stride[1]
        if colrem: colpad = stride[1] - colrem
        pad = ((pad, pad + rowpad), (pad, pad + colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn


def pool_fn(pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    if mode == 'avg': mode = 'average_exc_pad'
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def decomp(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def addtrans_decomp(X, Y=None, splits=5):
    if Y is None: Y = X
    size = X.shape[0]
    starts = decomp(size, splits)
    result = None
    for i in xrange(splits):
        Xtmp = X[starts[i]:starts[i + 1], :] + Y[:, starts[i]:starts[i + 1]].T
        result = np.concatenate([result, Xtmp], axis=0) if result is not None else Xtmp
    return result


def kernel(Xtr, Xte=None, kernel_type='rbf', kernel_args=(1.,)):
    rows_tr = Xtr.shape[0]
    if not isinstance(kernel_args, (tuple, list)): kernel_args = (kernel_args,)
    if kernel_type == 'rbf':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            omega = addtrans_decomp(H) - 2 * np.dot(Xtr, Xtr.T)
            del H, Xtr
            omega = np.exp(-omega / kernel_args[0])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            omega = addtrans_decomp(Htr, Hte) - 2 * np.dot(Xtr, Xte.T)
            del Htr, Hte, Xtr, Xte
            omega = np.exp(-omega / kernel_args[0])
    elif kernel_type == 'lin':
        if Xte is None:
            omega = np.dot(Xtr, Xtr.T)
        else:
            omega = np.dot(Xtr, Xte.T)
    elif kernel_type == 'poly':
        if Xte is None:
            omega = (np.dot(Xtr, Xtr.T) + kernel_args[0]) ** kernel_args[1]
        else:
            omega = (np.dot(Xtr, Xte.T) + kernel_args[0]) ** kernel_args[1]
    elif kernel_type == 'wav':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            H1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = H1 - H1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            Htr1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_te, axis=1)
            Hte1 = np.repeat(np.sum(Xte, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = Htr1 - Hte1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
    else:
        raise NotImplementedError
    return omega


class KELM(object):
    def __init__(self, C, kernel_type, kernel_args, beta=None):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_args = kernel_args
        self.beta = beta

    def train(self, inputX, inputy=None):
        self.trainX = inputX
        omega = kernel(inputX, None, self.kernel_type, self.kernel_args)
        rows = omega.shape[0]
        Crand = abs(np.random.uniform(0.1, 1.1)) * self.C
        self.beta = solve(np.eye(rows) / Crand + omega, inputy)
        out = np.dot(omega, self.beta)
        return out

    def predict(self, inputX):
        omega = kernel(self.trainX, inputX, self.kernel_type, self.kernel_args)
        del inputX
        out = np.dot(omega.T, self.beta)
        return out

    def load_trainX(self, trainX):
        self.trainX = trainX


class KELMcv(object):
    def __init__(self, C_range, kernel_type, kernel_args_list):
        self.C_range = C_range
        self.kernel_type = kernel_type
        self.kernel_args_list = kernel_args_list

    def train_cv(self, inputX, inputy):
        self.trainX = inputX
        self.beta_list = []
        optacc = 0.
        optC = None
        optarg = None
        for kernel_args in self.kernel_args_list:
            omega = kernel(inputX, None, self.kernel_type, kernel_args)
            rows = omega.shape[0]
            for C in self.C_range:
                Crand = C
                beta = solve(np.eye(rows) / Crand + omega, inputy)
                out = np.dot(omega, beta)
                acc = accuracy(out, inputy)
                p, r = precision_recall(out, inputy)
                self.beta_list.append(copy(beta))
                print '\t', kernel_args, C, acc, p, r
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
        print 'train opt', optarg, optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        optbeta = None
        optarg = None
        num = 0
        for kernel_args in self.kernel_args_list:
            omega = kernel(self.trainX, inputX, self.kernel_type, kernel_args)
            for C in self.C_range:
                out = np.dot(omega.T, self.beta_list[num])
                acc = accuracy(out, inputy)
                p, r = precision_recall(out, inputy)
                print '\t', kernel_args, C, acc, p, r
                num += 1
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optbeta = self.beta_list[num]
                    optarg = kernel_args
            del omega
        print 'test opt', optarg, optC, optacc
        optclf = KELM(optC, self.kernel_type, optarg, beta=optbeta)
        return optclf


class SVMlin(object):
    def __init__(self, C, tol):
        self.C = C
        self.tol = tol

    def train(self, inputX, inputy):
        dual = inputX.shape[0] < inputX.shape[1]
        self.clf = svm.LinearSVC(C=self.C, dual=dual, max_iter=10 ** 4, penalty='l2', loss='squared_hinge',
                                 tol=self.tol, multi_class='ovr', fit_intercept=True, intercept_scaling=1)
        # self.clf = svm.SVC(C=self.C, kernel='linear')
        self.clf.fit(inputX, inputy)
        return self.clf.predict(inputX)

    def predict(self, inputX):
        return self.clf.predict(inputX)

    def load_clf(self, clf):
        self.clf = clf


class SVMlincv(object):
    def __init__(self, C_range, tol):
        self.C_range = C_range
        self.tol = tol

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        self.clf_list = []
        for C in self.C_range:
            clf = SVMlin(C, self.tol)
            out = clf.train(inputX, inputy)
            acc = accuracy(out, inputy)
            p, r = precision_recall(out, inputy)
            self.clf_list.append(deepcopy(clf))
            print '\t', C, acc, p, r
            if acc > optacc:
                optacc = acc
                optC = C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            out = clf.predict(inputX)
            acc = accuracy(out, inputy)
            p, r = precision_recall(out, inputy)
            print '\t', clf.C, acc, p, r
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc
        optclf = SVMlin(optC, self.tol)
        return optclf

