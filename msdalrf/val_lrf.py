# coding:utf-8

import time
from collections import OrderedDict
import cPickle
import numpy as np
import pre
from msdalrf.clf import *
from msdalrf.layer_lrf_decomp import *


class mSDAELM(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # layer1
        # net['layer1'] = LRFLayer(dir_name='lrf_layer1', C=self.C, n_hidden=32, fsize=5,
        #                          pad=2, stride=1, pad_=0, stride_=1, noise=0.5,
        #                          pool_size=2, mode='max', add_pool=False, visual=True)
        # net['layer1'] = LRFLayer_chs(dir_name='lrf_layer1', C=self.C, n_hidden=32, fsize=5,
        #                              pad=2, stride=1, pad_=0, stride_=1, noise=0.25,
        #                              pool_size=None, mode=None, add_pool=False, visual=True)
        # layer2
        # net['layer2'] = LRFLayer(dir_name='lrf_layer2', C=self.C, n_hidden=32, fsize=5,
        #                          pad=2, stride=1, pad_=0, stride_=1, noise=0.5,
        #                          pool_size=7, mode='max', add_pool=True, visual=True)
        # net['decomp'] = DecompLayer(dir_name='lrf_layer1', C=self.C,
        #                             n_hidden1=32, fsize1=5, pad1=2, stride1=1, pad1_=0, stride1_=1, noise1=0.25,
        #                             n_hidden2=32, fsize2=5, pad2=2, stride2=1, pad2_=0, stride2_=1, noise2=0.25,
        #                             method='ELMAE', pool_size=8, mode='max', visual=False)
        net['decomp'] = DecompLayer_chs(dir_name='lrf_layer1', C=self.C,
                                        n_hidden1=1024, fsize1=5, pad1=2, stride1=1, pad1_=0, stride1_=3, noise1=0.3,
                                        n_hidden2=64, fsize2=5, pad2=2, stride2=1, pad2_=0, stride2_=3, noise2=0.3,
                                        method='ELMAE', add_pool1=False, pool_size1=2, mode1='max',
                                        add_pool2=True, pool_size2=10, mode2='avg', shortcut=1., visual=False)
        return net

    def _get_train_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            print out.shape
            out = layer.get_train_output_for(out)
            print 'add ' + name,
        print
        return out

    def _get_test_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            print out.shape
            out = layer.get_test_output_for(out)
            print 'add ' + name,
        print
        return out

    def train(self, inputX, name, unlabelX=None):
        nlabeled = inputX.shape[0]
        if unlabelX is not None:
            inputX = np.concatenate((inputX, unlabelX), axis=0)
        self.net = self._build()
        netout = self._get_train_output(self.net, inputX)
        print netout.shape
        netout = netout.reshape((netout.shape[0], -1))
        netout = netout[:nlabeled]
        np.save(name, netout)

    def test(self, inputX, name):
        netout = self._get_test_output(self.net, inputX)
        print netout.shape
        netout = netout.reshape((netout.shape[0], -1))
        np.save(name, netout)


def train_clf(tr_X, tr_y):
    tr_X = tr_X.reshape((tr_X.shape[0], -1))
    # clf = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 1., 0.5), tol=10 ** -4, jobs=8)
    clf = Classifier_KELMcv(C_range=10 ** np.arange(0., 4., 1.), kernel_type='rbf',
                            kernel_args_list=10 ** np.arange(1., 3., 0.5))
    clf.train_cv(tr_X, tr_y)
    return clf


def test_clf(clf, te_X, te_y):
    te_X = te_X.reshape((te_X.shape[0], -1))
    clf.test_cv(te_X, te_y)


def main():
    tr_X, tr_y = pre.load_train(size='_t')
    tr_X = pre.norm4d_per_sample(tr_X)
    tr_y = pre.one_hot(tr_y, 2)
    model = mSDAELM(C=1e5)
    model.train(tr_X, 'tr_netout.npy', )
    del tr_X
    te_X, te_y = pre.load_test(size='_t')
    te_X = pre.norm4d_per_sample(te_X)
    te_y = pre.one_hot(te_y, 2)
    model.test(te_X, 'te_netout.npy')
    del te_X
    tr_X = np.load('tr_netout.npy')
    clf = train_clf(tr_X, tr_y)
    te_X = np.load('te_netout.npy')
    test_clf(clf, te_X, te_y)
    print time.asctime()


def main2():
    with open('model.pkl', 'r') as f:
        model = cPickle.load(f)
    inputX = np.load('')
    model.test(inputX, 'mid_netout.npy')
    mid_netout = np.load('mid_netout.npy')
    out = predict(mid_netout)


if __name__ == '__main__':
    main()
