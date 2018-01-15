# coding:utf-8
import numpy as np
from matplotlib import pylab
from PIL import Image
import os
import time

username = 'fh'
interpolation = 'nearest'
img_count = 0

__all__ = ['save_beta_fc', 'save_beta_lrf', 'save_beta_lrfchs', 'save_map_lrf']


def addPad(map2d, padWidth):
    row, col = map2d.shape
    hPad = np.zeros((row, padWidth))
    map2d = np.hstack((hPad, map2d, hPad))
    vPad = np.zeros((padWidth, col + 2 * padWidth))
    map2d = np.vstack((vPad, map2d, vPad))
    return map2d


def squareStack(map3d):
    mapNum = map3d.shape[0]
    row, col = map3d.shape[1:]
    side = int(np.ceil(np.sqrt(mapNum)))
    lack = side ** 2 - mapNum
    map3d = np.vstack((map3d, np.zeros((lack, row, col))))
    map2ds = [addPad(map3d[i], 1) for i in range(side ** 2)]
    return np.vstack([np.hstack(map2ds[i:i + side])
                      for i in range(0, side ** 2, side)])


def save_beta_fc(beta):
    save_path = os.path.join('/home', username, 'images', time.asctime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n_feature, n_hidden = beta.shape
    num = int(np.ceil(n_hidden / 100.))
    img_side = int(np.sqrt(n_feature))
    img = beta.T.reshape((-1, img_side, img_side))
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        one_img = img[:100]
        img = img[100:]
        pylab.imshow(squareStack(one_img), interpolation=interpolation)
        pic_path = os.path.join(save_path, str(i) + '.png')
        pylab.savefig(pic_path)
    pylab.close()


def save_beta_lrf(beta, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    beta = beta.T.reshape((channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()


def save_beta_lrfchs(beta, chs, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    size, channels = beta.shape
    size /= chs
    beta = np.split(beta.T, chs, axis=1)
    beta = np.concatenate(beta, axis=0)
    beta = beta.reshape((chs * channels, int(np.sqrt(size)), int(np.sqrt(size))))
    pylab.figure()
    pylab.gray()
    pylab.imshow(squareStack(beta), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()


def save_map_lrf(map, dir, name):
    global img_count
    save_path = os.path.join('/home', username, 'images', dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pic_path = os.path.join(save_path, str(img_count) + name + '.png')
    img_count += 1
    num = len(map)
    pylab.figure()
    pylab.gray()
    for i in xrange(num):
        pylab.subplot(1, num, i + 1)
        pylab.imshow(squareStack(map[i]), interpolation=interpolation)
    pylab.savefig(pic_path)
    pylab.close()


def save_beta_formal(beta, chs=3):
    save_path = os.path.join('/home', username, 'images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    size, channels = beta.shape
    size /= chs
    beta_chs = np.split(beta.T, chs, axis=1)
    num = int(np.ceil(channels / 225.))
    for i in xrange(num):
        plot_array = None
        for c in xrange(chs):
            part_array = beta_chs[c][:225]
            beta_chs[c] = beta_chs[c][225:]
            part_array = tile_raster_images(part_array, img_shape=(5, 5),
                                            tile_shape=(15, 15), tile_spacing=(1, 1))
            plot_array = np.concatenate((plot_array, part_array[None, :, :]), axis=0) \
                if plot_array is not None else part_array[None, :, :]
        plot_array = plot_array.transpose((1, 2, 0))
        pylab.imshow(plot_array, interpolation='nearest')
        pylab.savefig(os.path.join(save_path, str(i) + '.png'))


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
