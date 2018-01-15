# coding:utf-8
'''
msda使用elm的随机投影和relu激活
'''
import numpy as np
from numpy.linalg import solve
from msdalrf.design_lrf import *
from msdalrf.util_lrf import *
from msdalrf.visual import *
from copy import copy, deepcopy
from multiprocessing import Pool
import time

__all__ = ['DecompLayer', 'DecompLayer_chs']

splits = 12


def deploy(batch, n_jobs):
    starts = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    starts[:batch % n_jobs] += 1
    return starts


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


def choose_method(patches, n_hidden, noise, C, method=None):
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
        patches = add_noise(patches, noise, splits=1)
        H = np.dot(patches, W)  # dot_decomp_dim1(patches, W, splits=1)
        H = relu(H, splits=1)
        Q = np.dot(H.T, H)  # dottrans_decomp(H.T, splits=(1, 1))
        P = np.dot(H.T, patches[:, :-1])  # dot_decomp_dim2(H.T, patches[:, :-1], splits=1)
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
        Q, P = collaborate_mn_jobs(patches, W, noise, splits=10)
    else:
        raise NotImplementedError
    reg = np.eye(n_hidden) / C
    reg[-1, -1] = 0.
    beta = solve(reg + Q, P)
    return beta.T


class BetaLayer(object):
    def __init__(self, dir_name, C, n_hidden, fsize, pad_, stride_, noise,
                 im2col_ignore_border, im2colfn=None, method=None, visual=False):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.im2col_ignore_border = im2col_ignore_border
        self.im2colfn = im2colfn
        self.method = method
        self.visual = visual

    def getbeta(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        if self.im2colfn is None:
            self.im2colfn = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                          pad=self.pad_, ignore_border=self.im2col_ignore_border)
        patches = self.im2colfn(inputX)
        del inputX
        patches = norm(patches)
        # patches, mean, P = whiten(patches)
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=self.method)
        if self.visual: save_beta_lrf(beta, self.dir_name, 'betatr')
        return beta


def job(obj, tmp):
    batch_tmp = tmp.shape[0]
    tmp = obj.im2colfn(tmp)
    tmp = norm(tmp)
    # tmp, _, _ = whiten(tmp, self.mean, self.P)
    tmp = np.dot(tmp, obj.beta)
    tmp = tmp.reshape((batch_tmp, obj.orows, obj.ocols, -1)).transpose((0, 3, 1, 2))
    tmp = relu(tmp)
    if obj.add_pool:
        tmp = obj.poolfn(tmp)
    return tmp


class ForwardLayer(Layer):
    def __init__(self, dir_name, n_hidden, fsize, pad, stride, beta, pool_size, mode, add_pool,
                 im2col_ignore_border, im2colfn=None, poolfn=None, visual=False):
        self.dir_name = dir_name
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.beta = beta
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.im2col_ignore_border = im2col_ignore_border
        self.im2colfn = im2colfn
        self.poolfn = poolfn
        self.visual = visual

    def forward_decomp(self, inputX, splits=1):
        assert inputX.ndim == 4
        batch_split = int(round(float(inputX.shape[0]) / splits))
        splits = int(np.ceil(float(inputX.shape[0]) / batch_split))
        patches = None
        for _ in xrange(splits):
            tmp = inputX[:batch_split]
            inputX = inputX[batch_split:]
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'in')
            batch_tmp = tmp.shape[0]
            tmp = self.im2colfn(tmp)
            # GCN normalization and whitening
            tmp = norm(tmp)
            # tmp, _, _ = whiten(tmp, self.mean, self.P)
            # mapping
            tmp = np.dot(tmp, self.beta)
            tmp = tmp.reshape((batch_tmp, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'raw')
            # activation
            tmp = relu(tmp)
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'relu')
            # pooling
            if self.add_pool:
                tmp = self.poolfn(tmp)
                if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'pool')
            patches = np.concatenate([patches, tmp], axis=0) if patches is not None else tmp
            self.visual = False
        return patches

    def forward_decomp_jobs(self, inputX, splits=1):
        assert inputX.ndim == 4
        starts = deploy(inputX.shape[0], splits)
        pool = Pool(processes=splits)
        jobs = []
        for i in xrange(splits):
            tmp = inputX[:starts[i]]
            inputX = inputX[starts[i]:]
            jobs.append(pool.apply_async(job, (self, tmp)))
            del tmp
        pool.close()
        pool.join()
        patches = None
        for one_job in jobs:
            tmp = one_job.get()
            patches = np.concatenate([patches, tmp], axis=0) if patches is not None else tmp
        return patches

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride,
                                ignore_border=self.im2col_ignore_border)
        self.orows, self.ocols = oshape[-2:]
        if self.im2colfn is None:
            self.im2colfn = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                          pad=self.pad, ignore_border=self.im2col_ignore_border)
        if self.add_pool and self.poolfn is None:
            self.poolfn = pool_fn(self.pool_size, mode=self.mode)
        patches = self.forward_decomp_jobs(inputX, splits=splits)
        return patches

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels == 1
        patches = self.forward_decomp_jobs(inputX, splits=splits)
        return patches


class DecompLayer(Layer):
    def __init__(self, dir_name, C,
                 n_hidden1, fsize1, pad1, stride1, pad1_, stride1_, noise1,
                 n_hidden2, fsize2, pad2, stride2, pad2_, stride2_, noise2,
                 method, pool_size, mode, visual):
        self.dir_name = dir_name
        self.C = C
        # first layer
        self.n_hidden1 = n_hidden1
        self.fsize1 = fsize1
        self.stride1 = stride1
        self.pad1 = pad1
        self.stride1_ = stride1_
        self.pad1_ = pad1_
        self.noise1 = noise1
        # second layer
        self.n_hidden2 = n_hidden2
        self.fsize2 = fsize2
        self.stride2 = stride2
        self.pad2 = pad2
        self.stride2_ = stride2_
        self.pad2_ = pad2_
        self.noise2 = noise2
        self.method = method
        self.pool_size = pool_size
        self.mode = mode
        self.visual = visual

    def get_train_output_for(self, inputX):
        batches1, channels1, rows1, cols1 = inputX.shape
        assert channels1 == 1
        beta_border = True
        forward_border = False
        im2colfn_beta1 = im2col_compfn((rows1, cols1), self.fsize1, stride=self.stride1_,
                                       pad=self.pad1_, ignore_border=beta_border)
        im2colfn_forward1 = im2col_compfn((rows1, cols1), self.fsize1, stride=self.stride1,
                                          pad=self.pad1, ignore_border=forward_border)
        im2colfn_beta2 = None
        im2colfn_forward2 = None
        poolfn2 = None
        betalayer1 = BetaLayer(self.dir_name, self.C, self.n_hidden1, self.fsize1,
                               self.pad1_, self.stride1_, self.noise1, im2col_ignore_border=beta_border,
                               im2colfn=im2colfn_beta1, method=self.method, visual=self.visual)
        beta1 = betalayer1.getbeta(inputX)
        self.forwardlayer1_list = []
        self.forwardlayer2_list = []
        output = None
        for i in xrange(self.n_hidden1):
            starti = time.time()
            forwardlayer1 = ForwardLayer(self.dir_name, self.n_hidden1, self.fsize1,
                                         self.pad1, self.stride1, beta1[:, i], None, None, False,
                                         im2col_ignore_border=forward_border, im2colfn=im2colfn_forward1,
                                         poolfn=None, visual=self.visual)
            onech_out1 = forwardlayer1.get_train_output_for(inputX)
            self.forwardlayer1_list.append(deepcopy(forwardlayer1))
            ###################### compile once #####################
            batches2, channels2, rows2, cols2 = onech_out1.shape
            if im2colfn_beta2 is None:
                im2colfn_beta2 = im2col_compfn((rows2, cols2), self.fsize2, stride=self.stride2_,
                                               pad=self.pad2_, ignore_border=beta_border)
            if im2colfn_forward2 is None:
                im2colfn_forward2 = im2col_compfn((rows2, cols2), self.fsize2, stride=self.stride2,
                                                  pad=self.pad2, ignore_border=forward_border)
            if poolfn2 is None:
                poolfn2 = pool_fn(self.pool_size, mode=self.mode)
            ##########################################################
            betalayer2 = BetaLayer(self.dir_name, self.C, self.n_hidden2, self.fsize2,
                                   self.pad2_, self.stride2_, self.noise2, im2col_ignore_border=beta_border,
                                   im2colfn=im2colfn_beta2, method=self.method, visual=self.visual)
            beta2 = betalayer2.getbeta(onech_out1)
            for j in xrange(self.n_hidden2):
                startj = time.time()
                forwardlayer2 = ForwardLayer(self.dir_name, self.n_hidden2, self.fsize2,
                                             self.pad2, self.stride2, beta2[:, j], self.pool_size, self.mode, True,
                                             im2col_ignore_border=forward_border, im2colfn=im2colfn_forward2,
                                             poolfn=poolfn2, visual=self.visual)
                onech_out2 = forwardlayer2.get_train_output_for(onech_out1)
                self.forwardlayer2_list.append(deepcopy(forwardlayer2))
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print '\tj:', j, time.time() - startj
            print 'i:', i, time.time() - starti
        return output

    def get_test_output_for(self, inputX):
        assert inputX.shape[1] == 1
        forwardlayer1_iter = iter(self.forwardlayer1_list)
        forwardlayer2_iter = iter(self.forwardlayer2_list)
        output = None
        for i in xrange(self.n_hidden1):
            starti = time.time()
            forwardlayer1 = forwardlayer1_iter.next()
            onech_out1 = forwardlayer1.get_test_output_for(inputX)
            for j in xrange(self.n_hidden2):
                startj = time.time()
                forwardlayer2 = forwardlayer2_iter.next()
                onech_out2 = forwardlayer2.get_test_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                print '\tj:', j, time.time() - startj
            print 'i:', i, time.time() - starti
        return output


########################################################################################################################


def im2col_catch_compiled(inputX, im2colfn):
    assert inputX.ndim == 4
    patches = None
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2colfn(inputX[:, [0], :, :])
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if patches is not None else patches1ch
    return patches


class BetaLayer_chs(object):
    def __init__(self, dir_name, C, n_hidden, fsize, pad_, stride_, noise,
                 im2col_ignore_border, im2colfn=None, method=None, visual=False):
        self.dir_name = dir_name
        self.C = C
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise = noise
        self.im2col_ignore_border = im2col_ignore_border
        self.im2colfn = im2colfn
        self.method = method
        self.visual = visual

    def getbeta(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        if self.im2colfn is None:
            self.im2colfn = im2col_compfn((rows, cols), self.fsize, stride=self.stride_,
                                          pad=self.pad_, ignore_border=self.im2col_ignore_border)
        patches = im2col_catch_compiled(inputX, self.im2colfn)
        patches = norm(patches)
        # patches, mean, P = whiten(patches)
        beta = choose_method(patches, self.n_hidden, self.noise, self.C, method=self.method)
        if self.visual: save_beta_lrfchs(beta, channels, self.dir_name, 'betatr')
        return beta


def job_chs(obj, tmp):
    batch_tmp = tmp.shape[0]
    tmp = im2col_catch_compiled(tmp, obj.im2colfn)
    tmp = norm(tmp)
    # tmp, _, _ = whiten(tmp, self.mean, self.P)
    tmp = np.dot(tmp, obj.beta)
    tmp = tmp.reshape((batch_tmp, obj.orows, obj.ocols, -1)).transpose((0, 3, 1, 2))
    tmp = relu(tmp)
    if obj.add_pool:
        tmp = obj.poolfn(tmp)
    return tmp


class ForwardLayer_chs(Layer):
    def __init__(self, dir_name, n_hidden, fsize, pad, stride, beta, pool_size, mode, add_pool,
                 im2col_ignore_border, im2colfn=None, poolfn=None, visual=False):
        self.dir_name = dir_name
        self.n_hidden = n_hidden
        self.fsize = fsize
        self.stride = stride
        self.pad = pad
        self.beta = beta
        self.pool_size = pool_size
        self.mode = mode
        self.add_pool = add_pool
        self.im2col_ignore_border = im2col_ignore_border
        self.im2colfn = im2colfn
        self.poolfn = poolfn
        self.visual = visual

    def forward_decomp(self, inputX, splits=1):
        assert inputX.ndim == 4
        batch_split = int(round(float(inputX.shape[0]) / splits))
        splits = int(np.ceil(float(inputX.shape[0]) / batch_split))
        patches = None
        for _ in xrange(splits):
            tmp = inputX[:batch_split]
            inputX = inputX[batch_split:]
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'in')
            batch_tmp = tmp.shape[0]
            tmp = im2col_catch_compiled(tmp, self.im2colfn)
            # GCN normalization and whitening
            tmp = norm(tmp)
            # tmp, _, _ = whiten(tmp, self.mean, self.P)
            # mapping
            tmp = np.dot(tmp, self.beta)
            tmp = tmp.reshape((batch_tmp, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'raw')
            # activation
            tmp = relu(tmp)
            if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'relu')
            # pooling
            if self.add_pool:
                tmp = self.poolfn(tmp)
                if self.visual: save_map_lrf(tmp[[10, 100, 1000]], self.dir_name, 'pool')
            patches = np.concatenate([patches, tmp], axis=0) if patches is not None else tmp
            self.visual = False
        return patches

    def forward_decomp_jobs(self, inputX, splits=1):
        assert inputX.ndim == 4
        starts = deploy(inputX.shape[0], splits)
        pool = Pool(processes=splits)
        jobs = []
        for i in xrange(splits):
            tmp = inputX[:starts[i]]
            inputX = inputX[starts[i]:]
            jobs.append(pool.apply_async(job_chs, (self, tmp)))
            del tmp
        pool.close()
        pool.join()
        patches = None
        for one_job in jobs:
            tmp = one_job.get()
            patches = np.concatenate([patches, tmp], axis=0) if patches is not None else tmp
        return patches

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        oshape = conv_out_shape((batches, channels, rows, cols),
                                (self.n_hidden, channels, self.fsize, self.fsize),
                                pad=self.pad, stride=self.stride, ignore_border=self.im2col_ignore_border)
        self.orows, self.ocols = oshape[-2:]
        if self.im2colfn is None:
            self.im2colfn = im2col_compfn((rows, cols), self.fsize, stride=self.stride,
                                          pad=self.pad, ignore_border=self.im2col_ignore_border)
        if self.add_pool and self.poolfn is None:
            self.poolfn = pool_fn(self.pool_size, mode=self.mode)
        patches = self.forward_decomp_jobs(inputX, splits=splits)
        return patches

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        assert channels > 1
        patches = self.forward_decomp_jobs(inputX, splits=splits)
        return patches


class DecompLayer_chs(Layer):
    def __init__(self, dir_name, C,
                 n_hidden1, fsize1, pad1, stride1, pad1_, stride1_, noise1,
                 n_hidden2, fsize2, pad2, stride2, pad2_, stride2_, noise2,
                 method, add_pool1, pool_size1, mode1, add_pool2, pool_size2, mode2,
                 shortcut, visual):
        self.dir_name = dir_name
        self.C = C
        # first layer
        self.n_hidden1 = n_hidden1
        self.fsize1 = fsize1
        self.stride1 = stride1
        self.pad1 = pad1
        self.stride1_ = stride1_
        self.pad1_ = pad1_
        self.noise1 = noise1
        # second layer
        self.n_hidden2 = n_hidden2
        self.fsize2 = fsize2
        self.stride2 = stride2
        self.pad2 = pad2
        self.stride2_ = stride2_
        self.pad2_ = pad2_
        self.noise2 = noise2
        self.method = method
        self.add_pool1 = add_pool1
        self.pool_size1 = pool_size1
        self.mode1 = mode1
        self.add_pool2 = add_pool2
        self.pool_size2 = pool_size2
        self.mode2 = mode2
        self.shortcut = shortcut
        self.visual = visual

    def get_train_output_for(self, inputX):
        batches1, channels1, rows1, cols1 = inputX.shape
        assert inputX.shape[1] > 1
        beta_border = True
        forward_border = False
        im2colfn_beta1 = im2col_compfn((rows1, cols1), self.fsize1, stride=self.stride1_,
                                       pad=self.pad1_, ignore_border=beta_border)
        im2colfn_forward1 = im2col_compfn((rows1, cols1), self.fsize1, stride=self.stride1,
                                          pad=self.pad1, ignore_border=forward_border)
        poolfn1 = pool_fn(self.pool_size1, mode=self.mode1)
        im2colfn_beta2 = None
        im2colfn_forward2 = None
        poolfn2 = pool_fn(self.pool_size2, mode=self.mode2)
        betalayer1 = BetaLayer_chs(self.dir_name, self.C, self.n_hidden1, self.fsize1,
                                   self.pad1_, self.stride1_, self.noise1, im2col_ignore_border=beta_border,
                                   im2colfn=im2colfn_beta1, method=self.method, visual=self.visual)
        beta1 = betalayer1.getbeta(inputX)
        self.forwardlayer1_list = []
        self.forwardlayer2_list = []
        output = None
        self.shortcut = int(np.round(self.n_hidden1 * self.shortcut))
        for i in xrange(self.n_hidden1):
            starti = time.time()
            forwardlayer1 = ForwardLayer_chs(self.dir_name, self.n_hidden1, self.fsize1,
                                             self.pad1, self.stride1, beta1[:, i], self.pool_size1, self.mode1,
                                             self.add_pool1,
                                             im2col_ignore_border=forward_border, im2colfn=im2colfn_forward1,
                                             poolfn=poolfn1, visual=self.visual)
            onech_out1 = forwardlayer1.get_train_output_for(inputX)
            self.forwardlayer1_list.append(deepcopy(forwardlayer1))
            if i < self.shortcut:
                shortcut_layer = ShortcutLayer(poolfn=poolfn2)
                onech_out2 = shortcut_layer.get_train_output_for(onech_out1)
                self.forwardlayer2_list.append(deepcopy(shortcut_layer))
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
            else:
                ###################### compile once #####################
                batches2, channels2, rows2, cols2 = onech_out1.shape
                if im2colfn_beta2 is None:
                    im2colfn_beta2 = im2col_compfn((rows2, cols2), self.fsize2, stride=self.stride2_,
                                                   pad=self.pad2_, ignore_border=beta_border)
                if im2colfn_forward2 is None:
                    im2colfn_forward2 = im2col_compfn((rows2, cols2), self.fsize2, stride=self.stride2,
                                                      pad=self.pad2, ignore_border=forward_border)
                ##########################################################

                betalayer2 = BetaLayer(self.dir_name, self.C, self.n_hidden2, self.fsize2,
                                       self.pad2_, self.stride2_, self.noise2, im2col_ignore_border=beta_border,
                                       im2colfn=im2colfn_beta2, method=self.method, visual=self.visual)
                beta2 = betalayer2.getbeta(onech_out1)
                for j in xrange(self.n_hidden2):
                    startj = time.time()
                    forwardlayer2 = ForwardLayer(self.dir_name, self.n_hidden2, self.fsize2,
                                                 self.pad2, self.stride2, beta2[:, j], self.pool_size2, self.mode2,
                                                 self.add_pool2,
                                                 im2col_ignore_border=forward_border, im2colfn=im2colfn_forward2,
                                                 poolfn=poolfn2, visual=self.visual)
                    onech_out2 = forwardlayer2.get_train_output_for(onech_out1)
                    self.forwardlayer2_list.append(deepcopy(forwardlayer2))
                    output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                    print '\tj:', j, time.time() - startj
            print 'i:', i, time.time() - starti
        return output

    def get_test_output_for(self, inputX):
        assert inputX.shape[1] > 1
        forwardlayer1_iter = iter(self.forwardlayer1_list)
        forwardlayer2_iter = iter(self.forwardlayer2_list)
        output = None
        for i in xrange(self.n_hidden1):
            starti = time.time()
            forwardlayer1 = forwardlayer1_iter.next()
            onech_out1 = forwardlayer1.get_test_output_for(inputX)
            if i < self.shortcut:
                shortcut_layer = forwardlayer2_iter.next()
                onech_out2 = shortcut_layer.get_test_output_for(onech_out1)
                output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
            else:
                for j in xrange(self.n_hidden2):
                    startj = time.time()
                    forwardlayer2 = forwardlayer2_iter.next()
                    onech_out2 = forwardlayer2.get_test_output_for(onech_out1)
                    output = np.concatenate([output, onech_out2], axis=1) if output is not None else onech_out2
                    print '\tj:', j, time.time() - startj
            print 'i:', i, time.time() - starti
        return output


def job_shortcut(obj, tmp):
    tmp = obj.poolfn(tmp)
    return tmp


class ShortcutLayer(Layer):
    def __init__(self, poolfn):
        self.poolfn = poolfn

    def decomp_jobs(self, inputX, splits=1):
        assert inputX.ndim == 4
        starts = deploy(inputX.shape[0], splits)
        pool = Pool(processes=splits)
        jobs = []
        for i in xrange(splits):
            tmp = inputX[:starts[i]]
            inputX = inputX[starts[i]:]
            jobs.append(pool.apply_async(job_shortcut, (self, tmp)))
            del tmp
        pool.close()
        pool.join()
        patches = None
        for one_job in jobs:
            tmp = one_job.get()
            patches = np.concatenate([patches, tmp], axis=0) if patches is not None else tmp
        return patches

    def get_train_output_for(self, inputX):
        inputX = self.decomp_jobs(inputX, splits=splits)
        return inputX

    def get_test_output_for(self, inputX):
        inputX = self.decomp_jobs(inputX, splits=splits)
        return inputX
