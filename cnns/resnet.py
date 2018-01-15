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
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

import myUtils
from msdalrf import pre
from msdalrf.clf import Classifier_KELMcv
from collections import OrderedDict

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    net = []
    net.append((
        names[0],
        ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                  flip_filters=False, nonlinearity=None) if use_bias
        else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                       flip_filters=False, nonlinearity=None)
    ))

    net.append((
        names[1],
        BatchNormLayer(net[-1][1])
    ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return OrderedDict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = OrderedDict()

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(layers.get_output_shape(incoming_layer)[1] * ratio_n_filter), 1, int(1.0 / ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        layers.get_output_shape(net[last_layer_name])[1] * upscale_factor, 1, 1, 0, nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(layers.get_output_shape(incoming_layer)[1] * 4 * ratio_n_filter), 1, int(1.0 / ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_model(input_var):
    net = OrderedDict()
    net['input'] = InputLayer((None, 3, 224, 224), input_var)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

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
        feat = self.network['pool5']
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
    pkl_file = 'resnet50.pkl'
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
    tr_feat = 'feat/tr_resnetfeat.npy'
    te_feat = 'feat/te_resnetfeat.npy'
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
    # main()
    X = T.tensor4('X')
    print build_model(X).keys()
