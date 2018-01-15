# coding:utf-8
import numpy as np
from scipy.linalg import orth
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.neighbours import images2neibs
from lasagne.theano_extensions.padding import pad as lasagnepad
from sklearn.feature_selection import SelectKBest

__all__ = ['normal_random_bscale', 'uniform_random_bscale', 'sparse_random_bscale',
           'orthonormalize',
           'dot_decomp_dim1', 'dot_decomp_dim2', 'dottrans_decomp',
           'relu',
           'add_noise',
           'norm', 'whiten',
           'im2col_compfn', 'conv_out_shape', 'pool_fn', 'pool_op',
           'feature_select']


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


def mean(X, W):
    n_sample = X.shape[0]
    rand_idx = np.random.randint(0, n_sample, 10000)
    X_hidden = np.dot(X[rand_idx, :-1], W)
    hidden_mean = np.mean(abs(X_hidden), axis=0)
    return hidden_mean


def normal_random_bscale(X, input_unit, hidden_unit, bscale):
    std = 1.
    W = np.random.normal(loc=0, scale=std, size=(input_unit - 1, hidden_unit))
    b = np.random.normal(loc=0, scale=std, size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


def uniform_random_bscale(X, input_unit, hidden_unit, bscale):
    ranges = 1.
    W = np.random.uniform(low=-ranges, high=ranges, size=(input_unit - 1, hidden_unit))
    b = np.random.uniform(low=-ranges, high=ranges, size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


def sparse_random_bscale(X, input_unit, hidden_unit, bscale):
    uniform = np.random.uniform(low=0., high=1., size=(input_unit - 1, hidden_unit))
    s = 3.
    neg = np.where(uniform < 1. / (2. * s))
    pos = np.where(uniform > (1. - 1. / (2. * s)))
    W = np.zeros_like(uniform, dtype=float)
    W[neg] = -np.sqrt(s / input_unit)
    W[pos] = np.sqrt(s / input_unit)
    b = np.random.uniform(low=-1., high=1., size=(1, hidden_unit))
    bscale = mean(X, W) / bscale
    b *= bscale
    return np.vstack((W, b))


########################################################################################################################


def orthonormalize(filters):
    ndim = filters.ndim
    if ndim != 2:
        filters = np.expand_dims(filters, axis=0)
    rows, cols = filters.shape
    if rows >= cols:
        orthonormal = orth(filters)
    else:
        orthonormal = orth(filters.T).T
    if ndim != 2:
        orthonormal = np.squeeze(orthonormal, axis=0)
    return orthonormal


########################################################################################################################


def relu(X, splits=1):
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        xtmp = X[starts[i]:starts[i + 1]]
        X[starts[i]:starts[i + 1]] = 0.5 * (xtmp + abs(xtmp))
    return X


########################################################################################################################


def add_noise(X, noise, splits=1):
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        xtmp = X[starts[i]:starts[i + 1]]
        X[starts[i]:starts[i + 1]] = add_mn(xtmp, noise)
    return X


def add_mn(X, percent=0.5):
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=float)
    return X * binomial


########################################################################################################################


def norm(X, reg=0.1, splits=1):
    raw_shape = X.shape
    if len(raw_shape) > 2:
        X = X.reshape((raw_shape[0], -1))
    X = norm2d(X, reg, splits)
    if len(raw_shape) > 2:
        X = X.reshape(raw_shape)
    return X


# 对每一个patch里的元素去均值归一化
def norm2d(X, reg=0.1, splits=1):
    size = X.shape[0]
    starts = deploy(size, splits)
    for i in xrange(splits):
        Xtmp = X[starts[i]:starts[i + 1]]
        mean = Xtmp.mean(axis=1)
        Xtmp -= mean[:, None]
        normalizer = np.sqrt((Xtmp ** 2).mean(axis=1) + reg)
        Xtmp /= normalizer[:, None]
        X[starts[i]:starts[i + 1]] = Xtmp
    return X


def whiten(X, mean=None, P=None):
    raw_shape = X.shape
    if len(raw_shape) > 2:
        X = X.reshape((raw_shape[0], -1))
    tup = whiten2d(X, mean, P)
    lst = list(tup)
    if len(raw_shape) > 2:
        lst[0] = lst[0].reshape(raw_shape)
    return lst


def whiten2d(X, mean=None, P=None):
    if mean is None or P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        reg = np.mean(D)
        # reg = 0.1
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
        P = abs(P)
        X = X.dot(P)
    else:
        X -= mean
        X = X.dot(P)
    return X, mean, P


########################################################################################################################


def pool_fn(pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    if mode == 'avg': mode = 'average_exc_pad'
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_op(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    pool = pool_fn(pool_size, ignore_border, stride, pad, mode)
    return pool(xnp)


########################################################################################################################


# 根据图像的行列尺寸编译im2col函数,之后直接使用函数即可,比每次都编译速度快很多
def im2col_compfn(shape, fsize, stride, pad, ignore_border=False):
    assert len(shape) == 2
    assert isinstance(pad, int)
    if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
    if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
    X = T.tensor4()
    if not ignore_border:  # 保持下和右的边界
        rows, cols = shape
        rows, cols = rows + 2 * pad, cols + 2 * pad
        rowpad = colpad = 0
        rowrem = (rows - fsize[0]) % stride[0]
        if rowrem: rowpad = stride[0] - rowrem
        colrem = (cols - fsize[1]) % stride[1]
        if colrem: colpad = stride[1] - colrem
        pad = ((pad, pad + rowpad), (pad, pad + colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn


def conv_out_shape(inshape, outshape, pad, stride, ignore_border=False):
    batch, channel, mrows, mcols = inshape
    channelout, channelin, frows, fcols = outshape
    assert channel == channelin
    if isinstance(pad, tuple):
        rowpad, colpad = pad
    else:
        rowpad = colpad = pad
    if isinstance(stride, tuple):
        rowstride, colstride = stride
    else:
        rowstride = colstride = stride
    mrows, mcols = mrows + 2 * rowpad, mcols + 2 * colpad
    if not ignore_border:  # 保持下和右的边界
        rowrem = (mrows - frows) % rowstride
        if rowrem: mrows += rowstride - rowrem
        colrem = (mcols - fcols) % colstride
        if colrem: mcols += colstride - colrem
    orow = (mrows - frows) // rowstride + 1
    ocol = (mcols - fcols) // colstride + 1
    return batch, channelout, orow, ocol


########################################################################################################################


# compute X dot X.T
def dottrans_decomp(X, splits=(1, 1)):
    rows, cols = X.shape
    rsplit, csplit = splits
    assert rows % rsplit == 0 and cols % csplit == 0
    prows = rows / rsplit
    pcols = cols / csplit
    out = np.zeros((rows, rows), dtype=float)
    for j in xrange(csplit):
        col_list = [X[i * prows:(i + 1) * prows, j * pcols:(j + 1) * pcols] for i in xrange(rsplit)]
        for i in xrange(rsplit):
            for k in xrange(i, rsplit):
                part_dot = np.dot(col_list[i], col_list[k].T)
                out[i * prows:(i + 1) * prows, k * prows:(k + 1) * prows] += part_dot
                if i != k:
                    out[k * prows:(k + 1) * prows, i * prows:(i + 1) * prows] += part_dot.T
    return out


# compute X dot Y
def dot_decomp_dim1(X, Y, splits=1):
    size = X.shape[0]
    batchSize = int(round(float(size) / splits))
    splits = int(np.ceil(float(size) / batchSize))
    result = None
    for i in xrange(splits):
        tmp = X[:batchSize]
        X = X[batchSize:]
        tmp = np.dot(tmp, Y)
        result = np.concatenate([result, tmp], axis=0) if result is not None else tmp
    return result


def dot_decomp_dim2(X, Y, splits=1):
    Xrows, Xcols = X.shape
    Yrows, Ycols = Y.shape
    assert Xcols == Yrows
    batchSize = int(round(float(Xcols) / splits))
    splits = int(np.ceil(float(Xcols) / batchSize))
    out = np.zeros((Xrows, Ycols), dtype=float)
    for i in xrange(splits):
        Xtmp = X[:, :batchSize]
        X = X[:, batchSize:]
        Ytmp = Y[:batchSize, :]
        Y = Y[batchSize:, :]
        part_dot = np.dot(Xtmp, Ytmp)
        out += part_dot
    return out


########################################################################################################################


def feature_select(X, y=None, selector=None, k=4096):
    if selector is not None and y is not None:
        if y.ndim == 2: y = np.argmax(y, axis=1)
        selector = SelectKBest(k=k)
        X = selector.fit_transform(X, y)
        return selector, X
    else:
        X = selector.transform(X)
        return X
