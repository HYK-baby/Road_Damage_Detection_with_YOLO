import itertools
import math
import numpy as np
import tensorflow as tf


def generate_grid2D(h, w, step=1):
    # (x, y)
    shift_1 = np.arange(0, h) * step
    shift_2 = np.arange(0, w) * step
    X, Y = tf.meshgrid(shift_2, shift_1)   # shift_1 first, output shape=(feat_h,feat_w)
    X = tf.expand_dims(X, axis=-1)
    Y = tf.expand_dims(Y, axis=-1)
    grid = tf.concat([X, Y], axis=-1)     # shape=(feat_h, feat_w, 2)
    return grid


@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32), tf.TensorSpec(shape=None, dtype=tf.int32)])
def generate_tf_grid2D(h, w):
    """
    h = 5, w = 4
    (x, y):
    [0 0] [1 0] [2 0] [3 0]
    [0 1] [1 1] [2 1] [3 1]
    .
    .
    .
    [0 4] [1 4] [2 4] [3 4]
    """
    shift_1 = tf.range(0, h)
    shift_2 = tf.range(0, w)
    X, Y = tf.meshgrid(shift_2, shift_1)   # shift_1 first, output shape=(feat_h,feat_w)
    X = tf.expand_dims(X, axis=-1)
    Y = tf.expand_dims(Y, axis=-1)
    grid = tf.concat([X, Y], axis=-1)     # shape=(feat_h, feat_w, 2)
    return grid


def generate_default_boxes(ratios, scales, fm_sizes):
    """ Generate default boxes in SSD
    Args:
        scales: boxes' size relative to image's size
        fm_sizes: sizes of feature maps
        ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    num_boxes_level = []
    for m, fm_size in enumerate(fm_sizes):
        num_boxes_level.append(len(ratios[m])*2 + 2)
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = np.array(default_boxes, dtype=np.float32)
    default_boxes = np.clip(default_boxes, 0.0, 1.0)
# float
    return default_boxes, num_boxes_level
