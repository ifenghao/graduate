# coding:utf-8
import gc, cPickle
import numpy as np
from numpy.linalg import solve
from copy import copy, deepcopy
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_recall_fscore_support
from multiprocessing import Pool, cpu_count
from util_lrf import deploy

__all__ = ['Layer', 'Classifier_SVMlincv', 'Classifier_SVMlincv_jobs', 'Classifier_KELMcv', 'predict']


########################################################################################################################


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class CVInner(object):
    def get_train_acc(self, inputX, inputy):
        raise NotImplementedError

    def get_test_acc(self, inputX, inputy):
        raise NotImplementedError


class CVOuter(object):
    def train_cv(self, inputX, inputy):
        raise NotImplementedError

    def test_cv(self, inputX, inputy):
        raise NotImplementedError


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
    return p[0], r[1]


########################################################################################################################


class Classifier_SVMlin(CVInner):
    def __init__(self, C, tol):
        self.C = C
        self.tol = tol

    def get_train_acc(self, inputX, inputy):
        dual = inputX.shape[0] < inputX.shape[1]
        self.clf = svm.LinearSVC(C=self.C, dual=dual, max_iter=10 ** 5, penalty='l2', loss='squared_hinge',
                                 tol=self.tol, multi_class='ovr', fit_intercept=True, intercept_scaling=1)
        # self.clf = svm.SVC(C=self.C, kernel='linear')
        self.clf.fit(inputX, inputy)
        pred_y = self.clf.predict(inputX)
        return accuracy_score(y_true=inputy, y_pred=pred_y)

    def get_test_acc(self, inputX, inputy):
        pred_y = self.clf.predict(inputX)
        return accuracy_score(y_true=inputy, y_pred=pred_y)


class Classifier_SVMlincv(CVOuter):
    def __init__(self, C_range, tol):
        self.C_range = C_range
        self.tol = tol

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        self.clf_list = []
        for C in self.C_range:
            clf = Classifier_SVMlin(C, self.tol)
            acc = clf.get_train_acc(inputX, inputy)
            self.clf_list.append(deepcopy(clf))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            acc = clf.get_test_acc(inputX, inputy)
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc


def train_lin_job(C, tol, inputX, inputy):
    clf = Classifier_SVMlin(C, tol)
    acc = clf.get_train_acc(inputX, inputy)
    return clf, acc


def test_lin_job(clf, inputX, inputy):
    acc = clf.get_test_acc(inputX, inputy)
    return clf, acc


class Classifier_SVMlincv_jobs(CVOuter):
    def __init__(self, C_range, tol, jobs):
        self.C_range = C_range
        self.tol = tol
        self.jobs = jobs

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=self.jobs)
        jobs = []
        for C in self.C_range:
            jobs.append(pool.apply_async(train_lin_job, (C, self.tol, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        self.clf_list = []
        for one_job in jobs:
            clf, acc = one_job.get()
            self.clf_list.append(deepcopy(clf))
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=self.jobs)
        jobs = []
        for clf in self.clf_list:
            jobs.append(pool.apply_async(test_lin_job, (clf, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        for one_job in jobs:
            clf, acc = one_job.get()
            print '\t', clf.C, acc
            if acc > optacc:
                optacc = acc
                optC = clf.C
        print 'test opt', optC, optacc


########################################################################################################################


class Classifier_SVMrbf(CVInner):
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma

    def get_train_acc(self, inputX, inputy):
        self.clf = svm.SVC(C=self.C, gamma=self.gamma, kernel='rbf')
        self.clf.fit(inputX, inputy)
        return self.clf.score(inputX, inputy)

    def get_test_acc(self, inputX, inputy):
        return self.clf.score(inputX, inputy)


class Classifier_SVMrbfcv(CVOuter):
    def __init__(self, C_range, gamma_range):
        self.C_range = C_range
        self.gamma_range = gamma_range

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        optgamma = None
        self.clf_list = []
        for gamma in self.gamma_range:
            for C in self.C_range:
                clf = Classifier_SVMrbf(C, gamma)
                acc = clf.get_train_acc(inputX, inputy)
                self.clf_list.append(deepcopy(clf))
                print '\t', gamma, C, acc
                if acc > optacc:
                    optacc = acc
                    optgamma = gamma
                    optC = C
        print 'train opt', optgamma, optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        optacc = 0.
        optC = None
        optgamma = None
        for clf in self.clf_list:
            acc = clf.get_test_acc(inputX, inputy)
            print '\t', clf.gamma, clf.C, acc
            if acc > optacc:
                optacc = acc
                optgamma = clf.gamma
                optC = clf.C
        print 'test opt', optgamma, optC, optacc


def train_rbf_job(C, gamma, inputX, inputy):
    clf = Classifier_SVMrbf(C, gamma)
    acc = clf.get_train_acc(inputX, inputy)
    return clf, acc


def test_rbf_job(clf, inputX, inputy):
    acc = clf.get_test_acc(inputX, inputy)
    return clf, acc


class Classifier_SVMrbfcv_jobs(CVOuter):
    def __init__(self, C_range, gamma_range):
        self.C_range = C_range
        self.gamma_range = gamma_range

    def train_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=cpu_count())
        jobs = []
        for gamma in self.gamma_range:
            for C in self.C_range:
                jobs.append(pool.apply_async(train_rbf_job, (C, gamma, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        optgamma = None
        self.clf_list = []
        for one_job in jobs:
            clf, acc = one_job.get()
            self.clf_list.append(deepcopy(clf))
            print '\t', clf.gamma, clf.C, acc
            if acc > optacc:
                optacc = acc
                optgamma = clf.gamma
                optC = clf.C
        print 'train opt', optgamma, optC, optacc

    def test_cv(self, inputX, inputy):
        if inputy.ndim == 2: inputy = np.argmax(inputy, axis=1)
        pool = Pool(processes=cpu_count())
        jobs = []
        for clf in self.clf_list:
            jobs.append(pool.apply_async(test_rbf_job, (clf, inputX, inputy)))
        pool.close()
        pool.join()
        optacc = 0.
        optC = None
        optgamma = None
        for one_job in jobs:
            clf, acc = one_job.get()
            print '\t', clf.gamma, clf.C, acc
            if acc > optacc:
                optacc = acc
                optgamma = clf.gamma
                optC = clf.C
        print 'test opt', optgamma, optC, optacc


########################################################################################################################


def addtrans_decomp(X, Y=None, splits=5):
    if Y is None: Y = X
    size = X.shape[0]
    starts = deploy(size, splits)
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


class Classifier_KELM(Layer):
    def __init__(self, C, kernel_type, kernel_args):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_args = kernel_args

    def get_train_output_for(self, inputX, inputy=None):
        self.trainX = inputX
        omega = kernel(inputX, None, self.kernel_type, self.kernel_args)
        rows = omega.shape[0]
        Crand = abs(np.random.uniform(0.1, 1.1)) * self.C
        self.beta = solve(np.eye(rows) / Crand + omega, inputy)
        out = np.dot(omega, self.beta)
        return out

    def get_test_output_for(self, inputX):
        omega = kernel(self.trainX, inputX, self.kernel_type, self.kernel_args)
        del inputX
        out = np.dot(omega.T, self.beta)
        return out


class Classifier_KELMcv(CVOuter):
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
                Crand = abs(np.random.uniform(0.1, 1.1)) * C
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
            gc.collect()
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
            gc.collect()
        print 'test opt', optarg, optC, optacc
        with open('clf_params.pkl', 'w') as f:
            cPickle.dump([optbeta, self.kernel_type, optarg], f)

# 预测结果
def predict(inputX):
    with open('clf_params.pkl', 'r') as f:
        optbeta, kernel_type, optarg = cPickle.load(f)
    trainX = np.load('tr_netout.npy')
    omega = kernel(trainX, inputX, kernel_type, optarg)
    del trainX, inputX
    out = np.dot(omega.T, optbeta)
    return out
