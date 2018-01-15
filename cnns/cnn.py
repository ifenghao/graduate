# coding:utf-8

import time
import numpy as np
import theano.tensor as T
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
import myUtils
from msdalrf import pre


class Model(object):
    def __init__(self, lr, C, momentum):
        self.lr = lr
        self.C = C
        self.momentum = momentum
        self.X = T.tensor4('X')
        self.y = T.ivector('y')
        network = self._build()
        self.params = layers.get_all_params(network, trainable=True)
        reg = regularization.regularize_network_params(network, regularization.l2)
        reg /= layers.helper.count_params(network)
        # 训练集
        self.yDropProb = layers.get_output(network)
        trCrossentropy = objectives.categorical_crossentropy(self.yDropProb, self.y)
        self.trCost = trCrossentropy.mean() + C * reg
        # 验证、测试集
        self.yFullProb = layers.get_output(network, deterministic=True)
        vateCrossentropy = objectives.categorical_crossentropy(self.yFullProb, self.y)
        self.vateCost = vateCrossentropy.mean() + C * reg
        # 训练函数，输入训练集，输出训练损失和误差
        updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, momentum)
        self.trainfn = myUtils.basic.makeFunc([self.X, self.y], [self.trCost, self.yDropProb], updatesDict)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        self.vatefn = myUtils.basic.makeFunc([self.X, self.y], [self.vateCost, self.yFullProb], None)

    def _build(self):
        layer = layers.InputLayer(shape=(None, 3, 112, 112), input_var=self.X)
        layer = layers.Conv2DLayer(layer, num_filters=64, filter_size=(5, 5), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.MaxPool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False)
        layer = layers.Conv2DLayer(layer, num_filters=64, filter_size=(5, 5), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.MaxPool2DLayer(layer, pool_size=(8, 8), stride=None, pad=(0, 0), ignore_border=False)
        layer = layers.flatten(layer, outdim=2)  # 不加入展开层也可以，DenseLayer自动展开
        layer = layers.DropoutLayer(layer, p=0.5)
        layer = layers.DenseLayer(layer, num_units=2048,
                                  W=init.GlorotUniform(), b=init.Constant(0.),
                                  nonlinearity=nonlinearities.rectify)
        layer = layers.DropoutLayer(layer, p=0.5)
        layer = layers.DenseLayer(layer, num_units=2,
                                  W=init.GlorotUniform(), b=init.Constant(0.),
                                  nonlinearity=nonlinearities.softmax)
        return layer

    def trainModel(self, tr_X, va_X, tr_y, va_y, batchSize=64, maxIter=300, verbose=True,
                   start=10, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
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


def main():
    tr_X, tr_y = pre.load_train(resized=False)
    tr_X, avg, var = pre.norm4d(tr_X)
    # tr_y = pre.one_hot(tr_y, 2)
    tr_va_split = int(tr_X.shape[0] * 0.7)
    tr_X, va_X = tr_X[:tr_va_split], tr_X[tr_va_split:]
    tr_y, va_y = tr_y[:tr_va_split], tr_y[tr_va_split:]
    model = Model(lr=0.01, C=1, momentum=0.9)
    model.trainModel(tr_X, va_X, tr_y, va_y)
    del tr_X, va_X, tr_y, va_y
    te_X, te_y = pre.load_test(resized=False)
    # te_y = pre.one_hot(te_y, 2)
    te_X = pre.norm4d(te_X, avg, var)
    model.testModel(te_X, te_y)


if __name__ == '__main__':
    main()
