import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import im2col_compfn
from utils import norm4d_per_sample
from utils import dataset_path
from utils import dice
from scipy.misc import imresize
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from skimage import filters
import pysvg.parser
import cairosvg
import os

path = '/home/zfh/'


class Model(object):
    def __init__(self, model_name):
        if model_name is 'plain_cnn':
            from plain_cnn import PlainCNN
            model_class = PlainCNN
        elif model_name is 'deep_cnn':
            from deep_cnn import DeepCNN
            model_class = DeepCNN
        elif model_name is 'rl':
            model_class = None
        else:
            raise NotImplementedError
        self.model = model_class(istrained=True)

    def predict(self, inputX):
        inputX = norm4d_per_sample(inputX)
        return self.model.predict(inputX)


def readTiff(file_path):
    im = Image.open(file_path)
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


def resize(img, size):
    if img.ndim == 3:
        img = img.transpose((1, 2, 0))
    img = imresize(img, size, interp='nearest')
    if img.ndim == 3:
        img = img.transpose((2, 0, 1))
    return img


def vaild_bound(min_row, min_col, max_row, max_col, imgsize):
    if min_row < 0:
        max_row -= min_row
        min_row = 0
    if min_col < 0:
        max_col -= min_col
        min_col = 0
    if max_row > imgsize[0]:
        min_row -= max_row - imgsize[0]
        max_row = imgsize[0]
    if max_col > imgsize[1]:
        min_col -= max_col - imgsize[1]
        max_col = imgsize[1]
    return min_row, min_col, max_row, max_col


def extract_rgbpatch(inputX, im2col_fn):
    if len(inputX.shape) == 3:
        inputX = inputX[:, None, :, :]
    inputX = im2col_fn(inputX)
    fsize = int(np.sqrt(inputX.shape[1]))
    inputX = inputX.reshape((3, -1, fsize, fsize)).transpose((1, 0, 2, 3))
    return inputX


def pad_img(X, value, pad=(0, 0, 0, 0)):
    in_shape = X.shape
    out_shape = (in_shape[0] + pad[0] + pad[1],
                 in_shape[1] + pad[2] + pad[3])
    output = np.ones(out_shape, X.dtype) * value
    indices = (slice(pad[0], in_shape[0] + pad[0]),
               slice(pad[2], in_shape[1] + pad[2]))
    output[indices] = X
    return output


def gauss_weight(shape):
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    g_sum = np.sum(g)
    return g / g_sum


def filter_region(region, th=0.02):
    region_area = np.prod(region.shape[-2:])
    region_l = label(region)
    region_props = regionprops(region_l)
    for prop in region_props:
        if prop.area < region_area * th:
            region[prop.coords] = 0
    return region


def cal_dice(img, svg, fsize, stride, im2col_fn, model, th=0.4):
    img_size = img.shape[1]
    rgbpatch = extract_rgbpatch(img, im2col_fn)
    patch_pred = model.predict(rgbpatch)
    pred_size = (img_size - fsize) / stride + 1
    patch_pred = patch_pred.reshape((pred_size, pred_size))
    region = np.zeros_like(patch_pred, np.uint8)
    ext = fsize / (8 * stride)
    for i in xrange(pred_size):
        for j in xrange(pred_size):
            bbox = vaild_bound(i - ext, j - ext, i + ext, j + ext, (pred_size, pred_size))
            neibs = patch_pred[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if np.count_nonzero(neibs > th) > np.prod(neibs.shape) / 2:
                region[i, j] = 1
    # region = dilation(region, square(5))
    # region = filter_region(region)
    svg = resize(svg, (pred_size, pred_size))
    return dice(region, svg)


def valiadte(model_name, fsize):
    # constants
    fsize_in = 256
    stride_in = 16
    img_size_in = 2048
    model = Model(model_name)
    img_size = img_size_in * fsize / fsize_in
    stride = img_size * stride_in / img_size_in
    im2col_fn = im2col_compfn((img_size, img_size), fsize, stride, 0, False)
    test_path = dataset_path + 'test/'
    img_list = np.load(dataset_path + 'te_idx.npy')
    dice_all = 0.
    for img_name in img_list:
        img = readTiff(test_path + img_name)
        img = resize(img, (img_size, img_size))
        svg = readSvg(test_path, img_name[:-5] + '.svg')
        dice_all += cal_dice(img, svg, fsize, stride, im2col_fn, model)
    dice_all /= len(img_list)
    print dice_all


def ret_img(img, fsize, stride, model, th=0.4):
    row, col = img.shape[-2:]
    im2col_fn = im2col_compfn((row, col), fsize, stride, 0, False)
    rgbpatch = extract_rgbpatch(img, im2col_fn)
    patch_pred = model.predict(rgbpatch)
    row_pred, col_pred = (row - fsize) / stride + 1, (col - fsize) / stride + 1
    patch_pred = patch_pred.reshape((row_pred, col_pred))
    region = np.zeros_like(patch_pred, np.uint8)
    heatmap = np.zeros_like(patch_pred, np.float32)
    ext = fsize / (8 * stride)
    normal_bbox_size = 4 * ext * ext
    gauss_normal = gauss_weight((2 * ext, 2 * ext))
    for i in xrange(row_pred):
        for j in xrange(col_pred):
            bbox = vaild_bound(i - ext, j - ext, i + ext, j + ext, (row_pred, col_pred))
            neibs = patch_pred[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) != normal_bbox_size:
                gauss = gauss_weight(neibs.shape)
            else:
                gauss = gauss_normal
            heatmap[i, j] = np.average(neibs, weights=gauss)
            if np.count_nonzero(neibs > th) > np.prod(neibs.shape) / 2:
                region[i, j] = 255
    # region = dilation(region, square(5))
    # region = filter_region(region)
    return region, heatmap


def detect(img_path, model_name, fsize, fig1, fig2):
    # constants
    fsize_in = 256
    stride_in = 16
    img = readTiff(img_path)
    row_in, col_in = img.shape[1:]
    row = row_in * fsize / fsize_in
    col = col_in * fsize / fsize_in
    stride = row * stride_in / row_in
    img = resize(img, (row, col))
    model = Model(model_name)
    region, heatmap = ret_img(img, fsize, stride, model)
    region = resize(region, (row_in, col_in))
    map_max, map_min = np.max(heatmap), np.min(heatmap)
    map_range = map_max - map_min
    heatmap = (heatmap - map_min) / map_range if map_range > 0 else np.zeros_like(heatmap)
    # plot
    region = filters.sobel(region)
    region = np.where(region > 0, 255, 0)
    region = dilation(region, square(8))
    region = 255 - region
    region = np.array(region, np.uint8)
    region = Image.fromarray(region)
    region.save(path + fig1)
    plt.figure(figsize=(10, 10))
    cax = plt.matshow(heatmap, fignum=1, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    plt.jet()
    plt.savefig(path + fig2)


if __name__ == '__main__':
    # detect('/home/zfh/test.tiff', 'plain_cnn', 64, 'region.png', 'heatmap.png')
    # valiadte('plain_cnn', 64)
    x = np.ones((2048, 2048))
    x[0:100, 200:400] = 0
    x[500:800, 0:200] = 0
    x[100:300, 80:100] = 0
    plt.subplot(211)
    plt.imshow(x)
    l = label(x)
    p = regionprops(l)
    print len(p)
    for i in p:
        x[i.coords] = 0
    plt.subplot(212)
    plt.imshow(x)
    plt.show()
