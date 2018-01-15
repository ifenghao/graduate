# coding:utf-8

import time
import cPickle
import os
import numpy as np
import theano.tensor as T
from lasagne import layers
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import ConcatLayer
from lasagne.nonlinearities import softmax, linear

import myUtils
from msdalrf import pre
from msdalrf.clf import Classifier_KELMcv


def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
    ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(
        net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(
        net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(
        net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=1000,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)
    return net


class Model(object):
    def __init__(self, lr, C, momentum):
        self.lr = lr
        self.C = C
        self.momentum = momentum
        self.X = T.tensor4('X')
        self.y = T.ivector('y')
        self.network = build_model(self.X)
        self.prob = self.network['prob']
        feat = self.network['pool5/7x7_s1']
        featout = layers.get_output(feat, deterministic=True)
        self.params = layers.get_all_params(self.prob, trainable=True)
        reg = regularization.regularize_network_params(self.prob, regularization.l2)
        reg /= layers.helper.count_params(self.prob)
        # 训练集
        self.yDropProb = layers.get_output(self.prob)
        trCrossentropy = objectives.categorical_crossentropy(self.yDropProb, self.y)
        self.trCost = trCrossentropy.mean() + C * reg
        # 验证、测试集
        self.yFullProb = layers.get_output(self.prob, deterministic=True)
        vateCrossentropy = objectives.categorical_crossentropy(self.yFullProb, self.y)
        self.vateCost = vateCrossentropy.mean() + C * reg
        # 训练函数，输入训练集，输出训练损失和误差
        updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, momentum)
        self.trainfn = myUtils.basic.makeFunc([self.X, self.y], [self.trCost, self.yDropProb], updatesDict)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        self.vatefn = myUtils.basic.makeFunc([self.X, self.y], [self.vateCost, self.yFullProb], None)
        # 输出特征
        self.featfn = myUtils.basic.makeFunc([self.X], [featout], None)

    def loadParams(self, vgg16params):
        layers.set_all_param_values(self.prob, vgg16params)

    def saveParams(self, filepath):
        valueList = layers.get_all_param_values(self.prob)
        cPickle.dump(valueList, open(os.path.join(filepath, 'vgg16.pkl'), 'w'))

    def trainModel(self, tr_X, va_X, tr_y, va_y, batchSize=64, maxIter=100, verbose=True,
                   start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        trainfn = self.trainfn
        validatefn = self.vatefn
        lr = self.lr

        earlyStop = myUtils.basic.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in xrange(maxIter):  # every epoch
            # In each epoch, we do a full pass over the training data:
            trAllPred = None
            trRandy = None
            trCostSum = 0.
            startTime = time.time()
            for batch in myUtils.basic.miniBatchGen(tr_X, tr_y, batchSize, shuffle=True):
                Xb, yb = batch
                trCost, trPred = trainfn(Xb, yb)
                trCostSum += trCost
                trAllPred = np.concatenate((trAllPred, trPred), axis=0) \
                    if trAllPred is not None else trPred
                trRandy = np.concatenate((trRandy, yb)) if trRandy is not None else yb
            trIter = len(tr_X) // batchSize
            if len(tr_X) % batchSize != 0: trIter += 1
            trCostMean = trCostSum / trIter
            trAcc = myUtils.basic.accuracy(trAllPred, trRandy)
            trP, trR = myUtils.basic.precision_recall(trAllPred, trRandy)
            # And a full pass over the validation data:
            vaAllPred = None
            vaCostSum = 0.
            for batch in myUtils.basic.miniBatchGen(va_X, va_y, batchSize, shuffle=False):
                Xb, yb = batch
                vaCost, vaPred = validatefn(Xb, yb)
                vaCostSum += vaCost
                vaAllPred = np.concatenate((vaAllPred, vaPred), axis=0) \
                    if vaAllPred is not None else vaPred
            vaIter = len(va_X) // batchSize
            if len(va_X) % batchSize != 0: vaIter += 1
            vaCostMean = vaCostSum / vaIter
            vaAcc = myUtils.basic.accuracy(vaAllPred, va_y)
            vaP, vaR = myUtils.basic.precision_recall(vaAllPred, va_y)
            if verbose:
                print 'epoch ', epoch, ' time: %.3f' % (time.time() - startTime),
                print 'trcost: %.5f  tracc: %.5f  trp: %.5f  trr: %.5f' % (trCostMean, trAcc, trP, trR),
                print 'vacost: %.5f  vaacc: %.5f  vap: %.5f  var: %.5f' % (vaCostMean, vaAcc, vaP, vaR)
            # Then we decide whether to early stop:
            if earlyStop.send((trCostMean, vaCostMean)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, self.momentum)
                trainfn = myUtils.basic.makeFunc([self.X, self.y], [self.trCost, self.yDropProb], updatesDict)
                totalStopCount += 1
                if totalStopCount > totalStopTol:  # 如果学习率降低仍然发生早停止，则退出迭代
                    print 'stop'
                    break
                if verbose: print 'learning rate decreases to ', lr

    def testModel(self, te_X, te_y, batchSize=64, verbose=True):
        testfn = self.vatefn
        teAllPred = None
        teCostSum = 0.
        for batch in myUtils.basic.miniBatchGen(te_X, te_y, batchSize, shuffle=False):
            Xb, yb = batch
            teCost, tePred = testfn(Xb, yb)
            teCostSum += teCost
            teAllPred = np.concatenate((teAllPred, tePred), axis=0) \
                if teAllPred is not None else tePred
        teIter = len(te_X) // batchSize
        if len(te_X) % batchSize != 0: teIter += 1
        teCostMean = teCostSum / teIter
        teAcc = myUtils.basic.accuracy(teAllPred, te_y)
        teP, teR = myUtils.basic.precision_recall(teAllPred, te_y)
        if verbose:
            print 'tecost: %.5f  teacc: %.5f  tep: %.5f  ter: %.5f' % (teCostMean, teAcc, teP, teR)

    def getFeature(self, X, batchSize=64):
        featfn = self.featfn
        size = X.shape[0]
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        feat = None
        for start, end in zip(startRange, endRange):
            tmp = X[start:end]
            tmp = featfn(tmp)[0]
            feat = np.concatenate((feat, tmp), axis=0) if feat is not None else tmp
        return feat


def load_pkl(path='/home/zfh/dataset/pretrained_models/'):
    pkl_file = 'blvc_googlenet.pkl'
    pkldict = cPickle.load(open(path + pkl_file))
    return pkldict['param values']


def preprocess(X, avg=None):
    if avg is None:
        avg = np.mean(X, axis=(0, 2, 3), dtype=np.float32, keepdims=True)
        return X - avg, avg
    else:
        return X - avg


def train_clf(tr_y, path, file):
    tr_X = np.load(path + file)
    tr_X = tr_X.reshape((tr_X.shape[0], -1))
    # clf = Classifier_SVMlincv_jobs(C_range=10 ** np.arange(-3., 1., 0.5), tol=10 ** -4, jobs=8)
    clf = Classifier_KELMcv(C_range=10 ** np.arange(0., 4., 1.), kernel_type='rbf',
                            kernel_args_list=10 ** np.arange(0., 4., 1.))
    clf.train_cv(tr_X, tr_y)
    return clf


def test_clf(clf, te_y, path, file):
    te_X = np.load(path + file)
    te_X = te_X.reshape((te_X.shape[0], -1))
    clf.test_cv(te_X, te_y)


def main():
    data_path = '/home/zfh/dataset/'
    tr_feat = 'feat/tr_googlenetfeat.npy'
    te_feat = 'feat/te_googlenetfeat.npy'
    params = load_pkl()
    model = Model(lr=0.01, C=1, momentum=0.9)
    model.loadParams(params)
    tr_X = np.load(data_path + 'tr_X_s.npy')
    tr_X, avg = preprocess(tr_X)
    feat = model.getFeature(tr_X)
    np.save(data_path + tr_feat, feat)
    te_X = np.load(data_path + 'te_X_s.npy')
    te_X = preprocess(te_X, avg)
    feat = model.getFeature(te_X)
    np.save(data_path + te_feat, feat)
    tr_y = np.load(data_path + 'tr_y.npy')
    te_y = np.load(data_path + 'te_y.npy')
    tr_y = pre.one_hot(tr_y, 2)
    te_y = pre.one_hot(te_y, 2)
    clf = train_clf(tr_y, data_path, tr_feat)
    test_clf(clf, te_y, data_path, te_feat)
    print time.asctime()


if __name__ == '__main__':
    main()
