# coding:utf-8

import time
from collections import OrderedDict
import cPickle
import numpy as np
from rllayer import DecompLayer_chs
from utils import dataset_path
from utils import load_train, load_test, norm4d_per_sample, one_hot
from utils import KELMcv, SVMlincv

feat_path = 'feat/'


def build_model():
    net = OrderedDict()
    # layer1
    # net['layer1'] = LRFLayer(dir_name='lrf_layer1', C=self.C, n_hidden=32, fsize=5,
    #                          pad=2, stride=1, pad_=0, stride_=1, noise=0.5,
    #                          pool_size=2, mode='max', add_pool=False, visual=True)
    net['decomp'] = DecompLayer_chs(dir_name='lrf_layer1', C=1e5,
                                    n_hidden1=64, fsize1=5, pad1=2, stride1=1, pad1_=0, stride1_=3, noise1=0.8,
                                    n_hidden2=64, fsize2=5, pad2=2, stride2=1, pad2_=0, stride2_=3, noise2=0.8,
                                    method='mLDEAE', add_pool1=False, pool_size1=2, mode1='max',
                                    add_pool2=True, pool_size2=40, mode2='avg', shortcut=0.5, visual=False)
    return net


class RL(object):
    def __init__(self, istrained, name, args=None):
        self.istrained = istrained
        self.name = name

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

    def train(self, tr_X, tr_y, te_X, te_y, un_X=None):
        nlabeled = tr_X.shape[0]
        if un_X is not None:
            tr_X = np.concatenate((tr_X, un_X), axis=0)
        net = build_model()
        trfeat = self._get_train_output(net, tr_X)
        print trfeat.shape
        trfeat = trfeat.reshape((trfeat.shape[0], -1))
        trfeat = trfeat[:nlabeled]
        cPickle.dump(net, open(dataset_path + self.name + '_net.pkl', 'w'))
        np.save(dataset_path + feat_path + self.name + 'trfeat.npy', trfeat)
        tefeat = self._get_test_output(net, te_X)
        print tefeat.shape
        tefeat = tefeat.reshape((tefeat.shape[0], -1))
        np.save(dataset_path + feat_path + self.name + 'tefeat.npy', tefeat)
        # clfcv = KELMcv(C_range=10 ** np.arange(0., 5., 0.5), kernel_type='rbf',
        #                kernel_args_list=10 ** np.arange(1., 6., 0.5))
        clfcv = SVMlincv(C_range=10 ** np.arange(0., 5., 0.5), tol=1e-2)
        clfcv.train_cv(trfeat, tr_y)
        optclf = clfcv.test_cv(tefeat, te_y)
        cPickle.dump(optclf, open(dataset_path + self.name + '_clf.pkl', 'w'))
        self.istrained = True

    def predict(self, X):
        assert self.istrained == True
        net = cPickle.load(open(dataset_path + self.name + '_net.pkl', 'r'))
        feat = self._get_test_output(net, X)
        print feat.shape
        feat = feat.reshape((feat.shape[0], -1))
        clf = cPickle.load(open(dataset_path + self.name + '_clf.pkl', 'r'))
        trfeat = np.load(dataset_path + feat_path + self.name + 'trfeat.npy')
        clf.load_trainX(trfeat)
        return clf.predict(feat)[:, 1]


def train():
    tr_X, tr_y = load_train(size='_t')
    tr_X = norm4d_per_sample(tr_X)
    tr_y = one_hot(tr_y, 2)
    te_X, te_y = load_test(size='_t')
    te_y = one_hot(te_y, 2)
    te_X = norm4d_per_sample(te_X)
    model = RL(istrained=False, name='rl_noun')
    model.train(tr_X, tr_y, te_X, te_y)


def predict():
    model = RL(istrained=True, name='rl_noun')
    te_X, _ = load_test(size='_t')
    te_X = norm4d_per_sample(te_X)
    return model.predict(te_X)


if __name__ == '__main__':
    train()
