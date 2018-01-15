# encoding=utf-8
from PIL import Image
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.neighbours import images2neibs
from lasagne.theano_extensions.padding import pad as lpad
import pysvg.parser
import cairosvg
import os


def readTiff(path, file_name):
    im = Image.open(path + file_name)
    im = np.array(im).transpose((2, 0, 1))
    return im


def readSvg(path, file_name):
    svg = pysvg.parser.parse(path + file_name)
    for ele in svg.getAllElements():
        ele.set_stroke('#000000')
        ele.set_fill('black')
    svg.save(path + 'tmp.svg')
    cairosvg.svg2png(url=path + 'tmp.svg', write_to=path + 'tmp.png')
    im = Image.open(path + 'tmp.png')
    im = np.array(im)[:, :, 3]
    im = np.where(im > 0, 1, 0)
    os.remove(path + 'tmp.svg')
    os.remove(path + 'tmp.png')
    return im


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
    Xpad = lpad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, mode='ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn


def extract_patch(inputX, im2col_fn):
    if len(inputX.shape) == 3:
        inputX = inputX[:, None, :, :]
    elif len(inputX.shape) == 2:
        inputX = inputX[None, None, :, :]
    inputX = im2col_fn(inputX)
    return inputX


def check_grid(inputX, im2col_fn, ratio=0.5):
    inputX = np.max(inputX, axis=1, keepdims=True)
    Xgrid = im2col_fn(inputX)
    n_grid = Xgrid.shape[0]
    Xavg = np.max(Xgrid, axis=1)
    n_obj = np.where(Xavg > 50)[0].shape[0]
    return n_obj > ratio * n_grid


def make_one_pos(img_path, tiff_name, svg_path, svg_name, im2col_dense, im2col_sparse, im2col_grid, ratio=0.6):
    img = 256 - readTiff(img_path, tiff_name)
    svg = readSvg(svg_path, svg_name)
    im2col = im2col_sparse if np.sum(svg) > svg.shape[0] * svg.shape[1] * 0.5 else im2col_dense
    img_patch = extract_patch(img, im2col)
    fsize = int(np.sqrt(img_patch.shape[1]))
    img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
    svg_patch = extract_patch(svg, im2col)
    area_obj = np.sum(svg_patch, axis=1)
    obj_idx = np.where(area_obj > ratio * fsize * fsize)[0]
    print len(obj_idx), tiff_name
    pos_patch = None
    for idx in obj_idx:
        if check_grid(img_patch[[idx]], im2col_grid):
            pos_patch = np.concatenate((pos_patch, img_patch[[idx]]), axis=0) \
                if pos_patch is not None else img_patch[[idx]]
    return pos_patch


def make_all_pos(path='/home/fh/dataset/', imgsize=2048, fsize=128, stride_dense=128, stride_sparse=256, grid=8):
    im2col_dense = im2col_compfn((imgsize, imgsize), fsize, stride_dense, 0, True)
    im2col_sparse = im2col_compfn((imgsize, imgsize), fsize, stride_sparse, 0, True)
    im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
    subset = 'cancer_subset0'
    n_subset = 8
    svg_path = path + 'labels/'
    for i in range(n_subset):
        subpath = path + subset + str(i) + '/'
        sublist = os.listdir(subpath)
        all_pos = None
        for img_name in sublist:
            svg_name = img_name[:-5] + '.svg'
            sub_pos = make_one_pos(subpath, img_name, svg_path, svg_name, im2col_dense, im2col_sparse, im2col_grid)
            if sub_pos is not None:
                all_pos = np.concatenate((all_pos, sub_pos), axis=0) \
                    if all_pos is not None else sub_pos
        np.save(path + 'pos' + str(i) + '.npy', all_pos)


def make_all_neg(path='/home/fh/dataset/', imgsize=2048, fsize=112, stride=128, grid=8, per_img=171):
    subpath = path + 'non_cancer_subset00/'
    im2col = im2col_compfn((imgsize, imgsize), fsize, stride, 0, True)
    im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
    sublist = os.listdir(subpath)
    all_neg = None
    cnt = 0
    for img_name in sublist:
        img = 256 - readTiff(subpath, img_name)
        img_patch = extract_patch(img, im2col)
        img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
        obj_list = []
        for i in range(img_patch.shape[0]):
            if check_grid(img_patch[[i]], im2col_grid, ratio=0.3):
                obj_list.append(i)
        img_patch = img_patch[obj_list]
        if img_patch.shape[0] > per_img:
            rand_idx = np.random.permutation(img_patch.shape[0])[:per_img]
            img_patch = img_patch[rand_idx]
        all_neg = np.concatenate((all_neg, img_patch), axis=0) \
            if all_neg is not None else img_patch
        cnt += 1
        if cnt % 20 == 0:
            np.save(path + 'neg' + str(cnt / 20 - 1) + '.npy', all_neg)
            all_neg = None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.misc import imresize

    make_all_pos()
    # img = np.load('/home/fh/dataset/pos0.npy')
    # print img.shape
    # for i in range(img.shape[0]):
    #     plt.imshow(img[i].transpose((1, 2, 0)))
    #     plt.show()
    # img = readTiff('/home/fh/', 'img.tiff')
    # plt.imshow(img.transpose((1,2,0)))
    # plt.show()
    # im2col = im2col_compfn((2048, 2048), 128, 128, 0, True)
    # img_patch = extract_patch(img, im2col)
    # img_patch = img_patch.reshape((3, -1, 128, 128)).transpose((1, 0, 2, 3))
    # for i in range(img_patch.shape[0]):
    #     tmp = img_patch[i].transpose((1, 2, 0))
    #     plt.subplot(211)
    #     plt.imshow(tmp)
    #     plt.subplot(212)
    #     plt.imshow(imresize(tmp, (224, 224)))
    #     plt.show()
