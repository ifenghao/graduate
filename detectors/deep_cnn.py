# coding:utf-8

import time
import cPickle
import numpy as np
import theano.tensor as T
from theano import function, In, Out
from lasagne import layers
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.nonlinearities import softmax, linear, rectify
from utils import load_train, load_test, norm4d_per_sample, one_hot
from utils import KELMcv
from utils import dataset_path

feat_path = 'feat/'


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


def build_vgg(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(
        net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(
        net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(
        net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


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


def build_google(input_var):
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

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

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


def build_res(input_var):
    net = {}
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


build_model_dict = {'vgg': build_vgg, 'google': build_google, 'res': build_res}
layer_dict = {'vgg': 'fc6', 'google': 'pool5/7x7_s1', 'res': 'pool5'}
pkl_dict = {'vgg': 'vgg19.pkl', 'google': 'blvc_googlenet.pkl', 'res': 'resnet50.pkl'}


class DeepCNN(object):
    def __init__(self, name, istrained=None, args=None):
        self.istrained = istrained
        self.cnn_name = name
        X = T.tensor4('X')
        build_model = build_model_dict[self.cnn_name]
        self.net = build_model(X)
        cnn_params = cPickle.load(open(dataset_path + 'pretrained_models/' + pkl_dict[name], 'r'))
        layers.set_all_param_values(self.net['prob'], cnn_params)
        feat = self.net[layer_dict[self.cnn_name]]
        featout = layers.get_output(feat, deterministic=True)
        self.featfn = makeFunc([X, ], [featout, ], None)

    def getFeature(self, X, batchSize=64):
        size = X.shape[0]
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        feat = None
        for start, end in zip(startRange, endRange):
            tmp = X[start:end]
            tmp = self.featfn(tmp)[0]
            feat = np.concatenate((feat, tmp), axis=0) if feat is not None else tmp
        return feat

    def train(self, tr_X, tr_y, te_X, te_y):
        trfeat = self.getFeature(tr_X)
        trfeat = trfeat.reshape((trfeat.shape[0], -1))
        np.save(dataset_path + feat_path + self.cnn_name + '_trfeat.npy', trfeat)
        tefeat = self.getFeature(te_X)
        tefeat = tefeat.reshape((tefeat.shape[0], -1))
        clfcv = KELMcv(C_range=10 ** np.arange(0., 5., 0.5), kernel_type='rbf',
                       kernel_args_list=10 ** np.arange(1., 6., 0.5))
        clfcv.train_cv(trfeat, tr_y)
        optclf = clfcv.test_cv(tefeat, te_y)
        cPickle.dump(optclf, open(dataset_path + feat_path + self.cnn_name + '_clf.npy', 'w'))
        self.istrained = True

    def predict(self, X):
        assert self.istrained == True
        feat = self.getFeature(X)
        feat = feat.reshape((feat.shape[0], -1))
        clf = cPickle.load(open(dataset_path + feat_path + self.cnn_name + '_clf.npy', 'r'))
        trfeat = np.load(dataset_path + feat_path + self.cnn_name + '_trfeat.npy')
        clf.load_trainX(trfeat)
        return clf.predict(feat)[:, 1]


def train():
    tr_X, tr_y = load_train(size='_t')
    tr_X = norm4d_per_sample(tr_X)
    tr_y = one_hot(tr_y, 2)
    te_X, te_y = load_test(size='_t')
    te_y = one_hot(te_y, 2)
    te_X = norm4d_per_sample(te_X)
    model = DeepCNN('vgg')
    model.train(tr_X, tr_y, te_X, te_y)


def predict():
    model = DeepCNN('vgg')
    te_X, _ = load_test(size='_t')
    te_X = norm4d_per_sample(te_X)
    return model.predict(te_X)


if __name__ == '__main__':
    train()
