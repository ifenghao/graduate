# coding:utf-8
'''
msda使用elm的随机投影和relu激活
'''
import numpy as np
from numpy.linalg import solve
from msdalrf.design_lrf import *
from msdalrf.util_lrf import *
from msdalrf.visual import *
from copy import copy

__all__ = ['LRFLayer', 'LRFLayer_chs']


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


def choose_method(patches, n_hidden, noise, C, method='ELMAE'):
    n_samples = patches.shape[0]
    bias = np.ones((n_samples, 1), dtype=float)
    patches = np.hstack((patches, bias))  # 最后一列偏置
    n_features = patches.shape[1]
    W = uniform_random_bscale(patches, n_features, n_hidden, 10.)
    # W = normal_random_bscale(X, n_features, self.n_hidden, 10.)
    # W = sparse_random_bscale(X, n_features, self.n_hidden, 10.)
    if method is None:
        return W[:-1, :]
    elif method == 'ELMAE':
        H = dot_decomp_dim1(patches, W, splits=10)
        H = relu(H)
        Q = dottrans_decomp(H.T, splits=(1, 10))
        P = dot_decomp_dim2(H.T, patches[:, :-1], splits=10)
    elif method == 'mLDEAE':
        S_X = dottrans_decomp(patches.T, splits=(1, 10))
        S_X_noise1 = add_Q_noise(copy(S_X), noise)
        Q = None
        left = np.dot(W.T, S_X_noise1)
        for i in xrange(n_hidden):
            right = np.dot(left, W[:, [i]])
            Q = np.concatenate((Q, right), axis=1) if Q is not None else right
        S_X_noise2 = add_P_noise(copy(S_X[:, :-1]), noise)  # 最后一列为偏置
        P = np.dot(W.T, S_X_noise2)
    elif method == 'mDEAE':
        Q, P = collaborate_mn(patches, W, noise, splits=100)
    else:
        raise NotImplementedError
    reg = np.eye(n_hidden) / C
    reg[-1, -1] = 0.
    beta = solve(reg + Q, P)
    return beta.T


class LRFLayer(Layer):
    def __init__(self, dir_name, C, n_hidden, fsize, pad, stride, pad_, stride_,
                 noise, pool_size, mode, add_pool, visual):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.visual = visual

    def _get_beta(self, onechX):
        assert onechX.ndim == 4 and onechX.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        # 将特征图转化为patch
        patches = self.im2colfn_getbeta(onechX)
        ##########################
        patches = norm(patches)
        # patches, self.mean1, self.P1 = whiten(patches)
        ##########################
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=None)
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_getbeta = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                              pad=self.pad_, ignore_border=True)
        self.im2colfn_forward = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        self.filters = []
        self.cccps = []
        output = None
        for ch in xrange(channels):
            onechX = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            if self.visual: save_map_lrf(onechX[[10, 100, 1000]], self.dir_name, 'intr')
            beta = self._get_beta(onechX)
            self.filters.append(copy(beta))
            if self.visual: save_beta_lrf(beta, self.dir_name, 'betatr')
            patches = self.im2colfn_forward(onechX)
            del onechX
            ##########################
            patches = norm(patches)
            # patches, _, _ = whiten(patches, self.mean1, self.P1)
            ##########################
            patches = dot_decomp_dim1(patches, beta, splits=10)
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawtr')
            # 归一化
            # patches = norm4d(patches)
            # save_map_lrf(patches[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            patches = relu(patches)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relutr')
            # 池化
            if self.add_pool:
                patches = pool_op(patches, self.pool_size, mode=self.mode)
                if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'pooltr')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if output is not None else patches
            print ch,
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        filter_iter = iter(self.filters)
        output = None
        for ch in xrange(channels):
            onechX = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            if self.visual: save_map_lrf(onechX[[10, 100, 1000]], self.dir_name, 'inte')
            patches = self.im2colfn_forward(onechX)
            del onechX
            ##########################
            patches = norm(patches)
            # patches, _, _ = whiten(patches, self.mean1, self.P1)
            ##########################
            patches = dot_decomp_dim1(patches, filter_iter.next(), splits=10)
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'rawte')
            # 归一化
            # patches = norm4d(patches)
            # save_map_lrf(patches[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            patches = relu(patches)
            if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'relute')
            # 池化
            if self.add_pool:
                patches = pool_op(patches, self.pool_size, mode=self.mode)
                if self.visual: save_map_lrf(patches[[10, 100, 1000]], self.dir_name, 'poolte')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if output is not None else patches
            print ch,
        return output


def im2col_catch_compiled(inputX, im2colfn):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2colfn(inputX[:, [0], :, :])
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


class LRFLayer_chs(Layer):
    def __init__(self, dir_name, C, n_hidden, fsize, pad, stride, pad_, stride_,
                 noise, pool_size, mode, add_pool, visual):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.visual = visual

    def _get_beta(self, inputX):
        assert inputX.ndim == 4
        patches = im2col_catch_compiled(inputX, self.im2colfn_getbeta)
        ##########################
        patches = norm(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=None)
        return beta

    def forward_decomp(self, inputX, beta):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / 10))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = None
        for _ in xrange(splits):
            patchestmp = im2col_catch_compiled(inputX[:batchSize], self.im2colfn_forward)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm(patchestmp)
            # patchestmp, _, _ = whiten(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if patches is not None else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'intr')
        batches, channels, rows, cols = inputX.shape
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.im2colfn_getbeta = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                              pad=self.pad_, ignore_border=True)
        self.im2colfn_forward = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                              pad=self.pad, ignore_border=False)
        # 学习beta
        self.beta = self._get_beta(inputX)
        if self.visual: save_beta_lrfchs(self.beta, channels, self.dir_name, 'beta')
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'rawtr')
        # 归一化
        # inputX = norm4d(inputX)
        # save_map_lrf(inputX[[10, 100, 1000]], dir_name, 'elmnorm')
        # 激活
        inputX = relu(inputX)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'relutr')
        # 池化
        if self.add_pool:
            inputX = pool_op(inputX, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'pooltr')
        return inputX

    def get_test_output_for(self, inputX):
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'inte')
        batches, channels, rows, cols = inputX.shape
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'rawte')
        # 归一化
        # inputX = norm4d(inputX)
        # save_map_lrf(inputX[[10, 100, 1000]], dir_name, 'elmnormte')
        # 激活
        inputX = relu(inputX)
        if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'relute')
        # 池化
        if self.add_pool:
            inputX = pool_op(inputX, self.pool_size, mode=self.mode)
            if self.visual: save_map_lrf(inputX[[10, 100, 1000]], self.dir_name, 'poolte')
        return inputX
