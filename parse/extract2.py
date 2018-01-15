# encoding=utf-8
from PIL import Image
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.neighbours import images2neibs
from lasagne.theano_extensions.padding import pad as lpad
from skimage.measure import label, regionprops
import pysvg.parser
import cairosvg
import os


def deploy(batch, n_jobs):
    dis_batch = (batch // n_jobs) * np.ones(n_jobs, dtype=np.int)
    dis_batch[:batch % n_jobs] += 1
    starts = np.cumsum(dis_batch)
    starts = [0] + starts.tolist()
    return starts


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


def check_grid(inputX, im2col_fn, ratio):
    inputX = np.max(inputX, axis=1, keepdims=True)
    Xgrid = im2col_fn(inputX)
    n_grid = Xgrid.shape[0]
    Xavg = np.max(Xgrid, axis=1)
    n_obj = np.where(Xavg > 50)[0].shape[0]
    return n_obj > ratio * n_grid


def choose_im2col(objarea, fsize, im2col_dict):
    farea = fsize * fsize
    times = float(objarea) / float(farea)
    if times < 4:
        return im2col_dict['4'], 2
    elif times < 8:
        return im2col_dict['8'], 4
    elif times < 16:
        return im2col_dict['16'], 6
    else:
        return im2col_dict['max'], 12


def vaild_bound(min_row, min_col, max_row, max_col, imgsize):
    if min_row < 0:
        max_row -= min_row
        min_row = 0
    if min_col < 0:
        max_col -= min_col
        min_col = 0
    if max_row > imgsize:
        min_row -= max_row - imgsize
        max_row = imgsize
    if max_col > imgsize:
        min_col -= max_col - imgsize
        max_col = imgsize
    return min_row, min_col, max_row, max_col


def extend_bbox(imgsize, fsize, bbox, extend=20):
    min_row, min_col, max_row, max_col = bbox
    min_row = min_row - extend if min_row - extend > 0 else 0
    min_col = min_col - extend if min_col - extend > 0 else 0
    max_row = max_row + extend if max_row + extend < imgsize else imgsize
    max_col = max_col + extend if max_col + extend < imgsize else imgsize
    row = max_row - min_row
    col = max_col - min_col
    if row < fsize:
        min_row -= (fsize - row) / 2
        max_row += (fsize - row) / 2
        if (fsize - row) % 2 == 1:
            max_row += 1
    if col < fsize:
        min_col -= (fsize - col) / 2
        max_col += (fsize - col) / 2
        if (fsize - col) % 2 == 1:
            max_col += 1
    return vaild_bound(min_row, min_col, max_row, max_col, imgsize)


def center_crop(img, prop, fsize, im2col_dict):
    imgsize = img.shape[1]
    crow, ccol = prop.centroid
    crow, ccol = int(np.round(crow)), int(np.round(ccol))
    min_row = crow - fsize / 2
    min_col = ccol - fsize / 2
    max_row = crow + fsize / 2
    max_col = ccol + fsize / 2
    if fsize % 2 == 1:
        max_row += 1
        max_col += 1
    min_row, min_col, max_row, max_col = vaild_bound(min_row, min_col, max_row, max_col, imgsize)
    one_patch = img[None, :, min_row:max_row, min_col:max_col]
    if check_grid(one_patch, im2col_dict['grid'], ratio=0.25):
        print 'center'
        return one_patch
    return None


def grid_crop(img, svg_label, prop, fsize, im2col_dict, ratio=0.33):
    objarea = 0.4 * (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1]) + 0.6 * prop.area
    im2col, per_max = choose_im2col(objarea, fsize, im2col_dict)
    bbox = extend_bbox(img.shape[1], fsize, prop.bbox)
    img_patch = extract_patch(img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]], im2col)
    img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
    svg_patch = extract_patch(svg_label[bbox[0]:bbox[2], bbox[1]:bbox[3]], im2col)
    area_obj = np.count_nonzero(svg_patch == prop.label, axis=1)
    obj_idx = np.where(area_obj > ratio * fsize * fsize)[0]
    print len(obj_idx)
    # if len(obj_idx) > per_max:
    #     rand_idx = np.random.permutation(len(obj_idx))[:per_max]
    #     obj_idx = obj_idx[rand_idx]
    # print len(obj_idx)
    one_patch = None
    for idx in obj_idx:
        if check_grid(img_patch[[idx]], im2col_dict['grid'], ratio=0.25):
            one_patch = np.concatenate((one_patch, img_patch[[idx]]), axis=0) \
                if one_patch is not None else img_patch[[idx]]
    return one_patch


def make_one_pos(img_path, tiff_name, svg_path, svg_name, fsize, im2col_dict):
    img = 256 - readTiff(img_path, tiff_name)
    svg = readSvg(svg_path, svg_name)
    svg_label = label(svg)
    svg_props = regionprops(svg_label)
    print tiff_name
    pos_patch = None
    for prop in svg_props:
        times = float((prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])) / float(fsize * fsize)
        if times <= 0.1:
            print 'ejected'
            one_patch = None
        elif times <= 1:
            one_patch = center_crop(img, prop, fsize, im2col_dict)
        else:
            one_patch = grid_crop(img, svg_label, prop, fsize, im2col_dict)
        if one_patch is not None:
            pos_patch = np.concatenate((pos_patch, one_patch), axis=0) \
                if pos_patch is not None else one_patch
    return pos_patch


# def make_all_pos(path='/home/fh/dataset/', imgsize=2048, fsize=80, grid=10):
#     im2col_dict = {}
#     im2col_dict['2'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 0.8, 0, True)
#     im2col_dict['4'] = im2col_compfn((imgsize, imgsize), fsize, fsize, 0, True)
#     im2col_dict['16'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 1.5, 0, True)
#     im2col_dict['64'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 2, 0, True)
#     im2col_dict['max'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 2.5, 0, True)
#     im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
#     subset = 'cancer_subset0'
#     n_subset = 9
#     svg_path = path + 'labels/'
#     for i in range(8, n_subset):
#         subpath = path + subset + str(i) + '/'
#         sublist = os.listdir(subpath)
#         all_pos = None
#         for img_name in sublist:
#             svg_name = img_name[:-5] + '.svg'
#             sub_pos = make_one_pos(subpath, img_name, svg_path, svg_name, fsize, im2col_dict, im2col_grid)
#             if sub_pos is not None:
#                 all_pos = np.concatenate((all_pos, sub_pos), axis=0) \
#                     if all_pos is not None else sub_pos
#         np.save(path + 'pos' + str(i) + '.npy', all_pos)


def make_all_pos(path='/home/fh/dataset/', imgsize=2048, fsize=256, grid=32, tr_te_ratio=0.75):
    im2col_dict = {}
    im2col_dict['4'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 0.75, 0, True)
    im2col_dict['8'] = im2col_compfn((imgsize, imgsize), fsize, fsize, 0, True)
    im2col_dict['16'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 1.25, 0, True)
    im2col_dict['max'] = im2col_compfn((imgsize, imgsize), fsize, fsize * 2, 0, True)
    im2col_dict['grid'] = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
    svg_path = path + 'labels/'
    subpath = path + 'cancer_subset00/'
    sublist = np.array(os.listdir(subpath))
    split = int(np.round(len(sublist) * tr_te_ratio))
    ridx = np.random.permutation(len(sublist))
    np.save(path + 'tr_posidx.npy', sublist[ridx[:split]])
    np.save(path + 'te_posidx.npy', sublist[ridx[split:]])
    for name, subl in zip(['tr', 'te'], [sublist[ridx[:split]], sublist[ridx[split:]]]):
        pos = None
        cnt = 0
        fcnt = 0
        starts = deploy(split, 5) if name is 'tr' else deploy(len(sublist) - split, 3)
        print starts
        for img_name in subl:
            svg_name = img_name[:-5] + '.svg'
            sub_pos = make_one_pos(subpath, img_name, svg_path, svg_name, fsize, im2col_dict)
            if sub_pos is not None:
                pos = np.concatenate((pos, sub_pos), axis=0) \
                    if pos is not None else sub_pos
            cnt += 1
            if cnt in starts:
                print cnt
                np.save(path + name + '_pos' + str(fcnt) + '.npy', pos)
                fcnt += 1
                pos = None


# def make_all_neg(path='/home/fh/dataset/', imgsize=2048, fsize=80, stride=128, grid=8, per_img=200):
#     subpath = path + 'non_cancer_subset00/'
#     im2col = im2col_compfn((imgsize, imgsize), fsize, stride, 0, True)
#     im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
#     sublist = os.listdir(subpath)
#     all_neg = None
#     cnt = 0
#     for img_name in sublist:
#         img = 256 - readTiff(subpath, img_name)
#         img_patch = extract_patch(img, im2col)
#         img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
#         obj_list = []
#         for i in range(img_patch.shape[0]):
#             if check_grid(img_patch[[i]], im2col_grid, ratio=0.25):
#                 obj_list.append(i)
#         img_patch = img_patch[obj_list]
#         if img_patch.shape[0] > per_img:
#             rand_idx = np.random.permutation(img_patch.shape[0])[:per_img]
#             img_patch = img_patch[rand_idx]
#         all_neg = np.concatenate((all_neg, img_patch), axis=0) \
#             if all_neg is not None else img_patch
#         cnt += 1
#         if cnt % 20 == 0:
#             np.save(path + 'neg' + str(cnt / 20 - 1) + '.npy', all_neg)
#             all_neg = None


def make_all_neg(path='/home/fh/dataset/', imgsize=2048, fsize=224, stride=200, grid=28, per_img=50, tr_te_ratio=0.75):
    subpath = path + 'non_cancer_subset00/'
    im2col = im2col_compfn((imgsize, imgsize), fsize, stride, 0, True)
    im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
    sublist = np.array(os.listdir(subpath))
    split = int(np.round(len(sublist)) * tr_te_ratio)
    ridx = np.random.permutation(len(sublist))
    np.save(path + 'tr_negidx.npy', sublist[ridx[:split]])
    np.save(path + 'te_negidx.npy', sublist[ridx[split:]])
    for name, subl in zip(['tr', 'te'], [sublist[ridx[:split]], sublist[ridx[split:]]]):
        neg = None
        cnt = 0
        fcnt = 0
        starts = deploy(split, 5) if name is 'tr' else deploy(len(sublist) - split, 3)
        print starts
        for img_name in subl:
            img = 256 - readTiff(subpath, img_name)
            img_patch = extract_patch(img, im2col)
            img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
            obj_list = []
            for i in range(img_patch.shape[0]):
                if check_grid(img_patch[[i]], im2col_grid, ratio=0.25):
                    obj_list.append(i)
            img_patch = img_patch[obj_list]
            if img_patch.shape[0] > per_img:
                rand_idx = np.random.permutation(img_patch.shape[0])[:per_img]
                img_patch = img_patch[rand_idx]
            neg = np.concatenate((neg, img_patch), axis=0) \
                if neg is not None else img_patch
            cnt += 1
            if cnt in starts:
                print cnt
                np.save(path + name + '_neg' + str(fcnt) + '.npy', neg)
                fcnt += 1
                neg = None


def make_all_unlabel(path='/home/fh/dataset/', imgsize=2048, fsize=224, stride=200, grid=28, per_img=40):
    subpath = path + 'unlabeled/'
    im2col = im2col_compfn((imgsize, imgsize), fsize, stride, 0, True)
    im2col_grid = im2col_compfn((fsize, fsize), fsize / grid, fsize / grid, 0, True)
    sublist = os.listdir(subpath)
    unl = None
    cnt = 0
    fcnt = 0
    starts = deploy(len(sublist), 10)
    print starts
    for img_name in sublist:
        img = 256 - readTiff(subpath, img_name)
        img_patch = extract_patch(img, im2col)
        img_patch = img_patch.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
        obj_list = []
        for i in range(img_patch.shape[0]):
            if check_grid(img_patch[[i]], im2col_grid, ratio=0.5):
                obj_list.append(i)
        img_patch = img_patch[obj_list]
        if img_patch.shape[0] > per_img:
            rand_idx = np.random.permutation(img_patch.shape[0])[:per_img]
            img_patch = img_patch[rand_idx]
        unl = np.concatenate((unl, img_patch), axis=0) \
            if unl is not None else img_patch
        cnt += 1
        if cnt in starts:
            print cnt
            np.save(path + 'unlabel' + str(fcnt) + '.npy', unl)
            fcnt += 1
            unl = None


def pos_img_counter(path='/home/fh/dataset/'):
    trcnt = 0
    tecnt = 0
    for i in range(5):
        a = np.load(path + 'tr_pos' + str(i) + '.npy')
        trcnt += len(a)
    for i in range(3):
        a = np.load(path + 'te_pos' + str(i) + '.npy')
        tecnt += len(a)
    print trcnt, tecnt, trcnt + tecnt  # 5197 1706 6903


def neg_img_counter(path='/home/fh/dataset/'):
    trcnt = 0
    tecnt = 0
    for i in range(5):
        a = np.load(path + 'tr_neg' + str(i) + '.npy')
        trcnt += len(a)
    for i in range(3):
        a = np.load(path + 'te_neg' + str(i) + '.npy')
        tecnt += len(a)
    print trcnt, tecnt, trcnt + tecnt  # 5188 1734 6922


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.misc import imresize

    make_all_pos()
    # make_all_neg()
    # make_all_unlabel()
    # pos_img_counter()
    # neg_img_counter()
    # img = np.load('/home/fh/dataset/tr_pos0.npy')
    # print img.shape
    # for i in range(img.shape[0]):
    #     print i
    #     plt.subplot(211)
    #     plt.bone()
    #     plt.imshow(img[i].transpose((1, 2, 0)))
    #     plt.subplot(212)
    #     plt.bone()
    #     plt.imshow(imresize(img[i].transpose((1, 2, 0)), (80, 80), interp='bilinear'))
    #     plt.show()
