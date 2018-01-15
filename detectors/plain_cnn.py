# coding:utf-8

import time
import cPickle
import numpy as np
import theano.tensor as T
from theano import function, In, Out
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from utils import accuracy, precision_recall
from utils import load_train, load_test, norm4d_per_sample
from utils import dataset_path


def makeFunc(inList, outList, updates):
    inputs = []
    for i in inList:
        inputs.append(In(i, borrow=True, allow_downcast=True))
    outputs = []
    for o in outList:
        outputs.append(Out(o, borrow=True))
    return function(
        inputs=inputs,
        outputs=outputs,
        updates=updates,
        allow_input_downcast=True
    )


def miniBatchGen(X, y, batchSize, shuffle=False):
    size = len(X)
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    for start, end in zip(startRange, endRange):
        excerpt = indices[start:end]
        if y is not None:
            yield X[excerpt], y[excerpt]
        else:
            yield X[excerpt]


def earlyStopGen(start=5, period=3, threshold=10, tol=2):
    trCostPeriod = []
    vaCostPeriod = []
    vaCostOpt = np.inf
    epoch = 0
    stopSign = False
    stopCount = 0
    while True:
        newCosts = (yield stopSign)  # 返回是否早停止
        epoch += 1
        if stopSign:  # 返回一个早停止标志后，重新检测
            stopSign = False
            stopCount = 0
        if epoch > start and newCosts is not None:  # send进来的元组在newCosts中
            trCost, vaCost = newCosts
            trCostPeriod.append(trCost)
            vaCostPeriod.append(vaCost)
            if vaCost < vaCostOpt:
                vaCostOpt = vaCost
            if len(trCostPeriod) >= period:
                P = np.mean(trCostPeriod) / np.min(trCostPeriod) - 1
                GL = np.mean(vaCostPeriod) / vaCostOpt - 1
                stopMetric = GL / P  # 停止的度量策略
                if stopMetric >= threshold:
                    stopCount += 1
                    if stopCount >= tol:
                        stopSign = True
                trCostPeriod = []  # 清空列表以继续判断下个周期
                vaCostPeriod = []


def build_model(input_var):
    layer = layers.InputLayer(shape=(None, 3, 224, 224), input_var=input_var)
    layer = layers.Conv2DLayer(layer, num_filters=64, filter_size=(3, 3), stride=(1, 1), pad='same')
    layer = layers.MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2), pad=(0, 0), ignore_border=False)
    layer = layers.Conv2DLayer(layer, num_filters=128, filter_size=(3, 3), stride=(1, 1), pad='same')
    layer = layers.MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2), pad=(0, 0), ignore_border=False)
    layer = layers.Conv2DLayer(layer, num_filters=256, filter_size=(3, 3), stride=(1, 1), pad='same')
    layer = layers.MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2), pad=(0, 0), ignore_border=False)
    layer = layers.Conv2DLayer(layer, num_filters=512, filter_size=(3, 3), stride=(1, 1), pad='same')
    layer = layers.MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2), pad=(0, 0), ignore_border=False)
    layer = layers.Conv2DLayer(layer, num_filters=512, filter_size=(3, 3), stride=(1, 1), pad='same')
    layer = layers.MaxPool2DLayer(layer, pool_size=(3, 3), stride=(2, 2), pad=(0, 0), ignore_border=False)
    layer = layers.flatten(layer, outdim=2)
    layer = layers.DenseLayer(layer, num_units=4096, nonlinearity=nonlinearities.rectify)
    layer = layers.DropoutLayer(layer, p=0.5)
    layer = layers.DenseLayer(layer, num_units=4096, nonlinearity=nonlinearities.rectify)
    layer = layers.DropoutLayer(layer, p=0.5)
    layer = layers.DenseLayer(layer, num_units=2, nonlinearity=nonlinearities.softmax)
    return layer


class PlainCNN(object):
    def __init__(self, istrained, name=None, args=None):
        self.istrained = istrained
        self.X = T.tensor4('X')
        self.y = T.ivector('y')
        self.outprob = build_model(self.X)
        if self.istrained:
            params = cPickle.load(open(dataset_path + 'plain_cnn.pkl', 'r'))
            layers.set_all_param_values(self.outprob, params)
            self.yFullProb = layers.get_output(self.outprob, deterministic=True)
            self.predfn = makeFunc([self.X, ], [self.yFullProb, ], None)
        else:
            self.lr, self.C, self.momentum = args
            self.params = layers.get_all_params(self.outprob, trainable=True)
            reg = regularization.regularize_network_params(self.outprob, regularization.l2)
            reg /= layers.helper.count_params(self.outprob)
            # 训练集
            self.yDropProb = layers.get_output(self.outprob)
            trCrossentropy = objectives.categorical_crossentropy(self.yDropProb, self.y)
            self.trCost = trCrossentropy.mean() + self.C * reg
            # 验证、测试集
            self.yFullProb = layers.get_output(self.outprob, deterministic=True)
            vateCrossentropy = objectives.categorical_crossentropy(self.yFullProb, self.y)
            self.vateCost = vateCrossentropy.mean() + self.C * reg
            # 训练函数，输入训练集，输出训练损失和误差
            updatesDict = updates.nesterov_momentum(self.trCost, self.params, self.lr, self.momentum)
            self.trainfn = makeFunc([self.X, self.y], [self.trCost, self.yDropProb], updatesDict)
            # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
            self.vatefn = makeFunc([self.X, self.y], [self.vateCost, self.yFullProb], None)

    def train(self, tr_X, tr_y, te_X, te_y, batchSize=32, maxIter=50,
              start=10, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        trainfn = self.trainfn
        lr = self.lr
        tr_va_split = int(tr_X.shape[0] * 0.7)
        tr_X, va_X = tr_X[:tr_va_split], tr_X[tr_va_split:]
        tr_y, va_y = tr_y[:tr_va_split], tr_y[tr_va_split:]

        earlyStop = earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in xrange(maxIter):  # every epoch
            # In each epoch, we do a full pass over the training data:
            trAllPred = None
            trRandy = None
            trCostSum = 0.
            startTime = time.time()
            for batch in miniBatchGen(tr_X, tr_y, batchSize, shuffle=True):
                Xb, yb = batch
                trCost, trPred = trainfn(Xb, yb)
                trCostSum += trCost
                trAllPred = np.concatenate((trAllPred, trPred), axis=0) \
                    if trAllPred is not None else trPred
                trRandy = np.concatenate((trRandy, yb)) if trRandy is not None else yb
            trIter = len(tr_X) // batchSize
            if len(tr_X) % batchSize != 0: trIter += 1
            trCostMean = trCostSum / trIter
            trAcc = accuracy(trAllPred, trRandy)
            trP, trR = precision_recall(trAllPred, trRandy)
            # And a full pass over the validation data:
            vaAllPred = None
            vaCostSum = 0.
            for batch in miniBatchGen(va_X, va_y, batchSize, shuffle=False):
                Xb, yb = batch
                vaCost, vaPred = self.vatefn(Xb, yb)
                vaCostSum += vaCost
                vaAllPred = np.concatenate((vaAllPred, vaPred), axis=0) \
                    if vaAllPred is not None else vaPred
            vaIter = len(va_X) // batchSize
            if len(va_X) % batchSize != 0: vaIter += 1
            vaCostMean = vaCostSum / vaIter
            vaAcc = accuracy(vaAllPred, va_y)
            vaP, vaR = precision_recall(vaAllPred, va_y)
            print 'epoch ', epoch, ' time: %.3f' % (time.time() - startTime),
            print 'trcost: %.5f  tracc: %.5f  trp: %.5f  trr: %.5f' % (trCostMean, trAcc, trP, trR),
            print 'vacost: %.5f  vaacc: %.5f  vap: %.5f  var: %.5f' % (vaCostMean, vaAcc, vaP, vaR)
            # Then we decide whether to early stop:
            if earlyStop.send((trCostMean, vaCostMean)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, self.momentum)
                trainfn = makeFunc([self.X, self.y], [self.trCost, self.yDropProb], updatesDict)
                totalStopCount += 1
                if totalStopCount > totalStopTol:  # 如果学习率降低仍然发生早停止，则退出迭代
                    print 'stop'
                    break
                print 'learning rate decreases to ', lr
        ################################################################################################################
        self.istrained = True
        params = layers.get_all_param_values(self.outprob)
        cPickle.dump(params, open(dataset_path + 'plain_cnn.pkl', 'w'))
        ################################################################################################################
        teAllPred = None
        teCostSum = 0.
        for batch in miniBatchGen(te_X, te_y, batchSize, shuffle=False):
            Xb, yb = batch
            teCost, tePred = self.vatefn(Xb, yb)
            teCostSum += teCost
            teAllPred = np.concatenate((teAllPred, tePred), axis=0) \
                if teAllPred is not None else tePred
        teIter = len(te_X) // batchSize
        if len(te_X) % batchSize != 0: teIter += 1
        teCostMean = teCostSum / teIter
        teAcc = accuracy(teAllPred, te_y)
        teP, teR = precision_recall(teAllPred, te_y)
        print 'tecost: %.5f  teacc: %.5f  tep: %.5f  ter: %.5f' % (teCostMean, teAcc, teP, teR)

    def predict(self, X, batchSize=32):
        assert self.istrained == True
        allPred = None
        for x in miniBatchGen(X, None, batchSize, shuffle=False):
            pred = self.predfn(x)[0]
            allPred = np.concatenate((allPred, pred), axis=0) \
                if allPred is not None else pred
        return allPred[:, 1]


def train():
    tr_X, tr_y = load_train(size='_t')
    tr_X = norm4d_per_sample(tr_X)
    te_X, te_y = load_test(size='_t')
    te_X = norm4d_per_sample(te_X)
    model = PlainCNN(istrained=False, args=(0.01, 0.1, 0.9))
    model.train(tr_X, tr_y, te_X, te_y)


def predict():
    model = PlainCNN(istrained=True)
    te_X, _ = load_test(size='_t')
    te_X = norm4d_per_sample(te_X)
    return model.predict(te_X)


if __name__ == '__main__':
    train()
