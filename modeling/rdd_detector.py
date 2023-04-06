import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, regularizers, initializers
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import backend as K
# from network_blocks import SE_attention, PAFPN, _conv_block, _dwconv_block, hard_swish, CSP_layer, _dwconv_dilation, Focus_block
from .network_blocks import SE_attention, PAFPN, _conv_block, _dwconv_block, hard_swish, CSP_layer, _dwconv_dilation, Focus_block


# Network20-2 with stem2 78.96%
# Network20-2 with stem1 78.17%
def my_stem_block2(inputs, num_init_features, weight_decay):
    # stem_block
    inter_channel = num_init_features//2
    feat_stem = _conv_block(inputs, inter_channel, name='stem', kernel_size=3, stride=2, pad='same', l2=weight_decay)
    # res_x = feat_stem
    feat_stem = layers.DepthwiseConv2D(3, padding='same', use_bias=False, name='stem1',
                                       depthwise_regularizer=regularizers.l2(weight_decay),
                                       depthwise_initializer=initializers.HeNormal())(feat_stem)
    feat_stem = layers.BatchNormalization(momentum=.99, name='stem1_BN')(feat_stem)
    feat_stem = _conv_block(feat_stem, inter_channel, name='stem1_pw', kernel_size=1, stride=1, l2=weight_decay)
    # feat_stem = feat_stem + res_x
    # feat_stem = _conv_block(feat_stem, num_init_features, name='stem1c', kernel_size=1, l2=weight_decay)
    feat_stem1 = _conv_block(feat_stem, inter_channel, name='stem1b', kernel_size=1, stride=1, l2=weight_decay)
    feat_stem2 = _conv_block(feat_stem, inter_channel, name='stem1c', kernel_size=1, stride=1, l2=weight_decay)
    # stem_branch1
    feat_stem1 = layers.DepthwiseConv2D(3, strides=(2, 2), padding='same', use_bias=False, name='stem1b_dw',
                                        depthwise_regularizer=regularizers.l2(weight_decay),
                                        depthwise_initializer=initializers.HeNormal())(feat_stem1)
    feat_stem1 = layers.BatchNormalization(momentum=.99, name='stem1b_dw_BN')(feat_stem1)
    feat_stem1 = _conv_block(feat_stem1, inter_channel, name='stem1b_pw', kernel_size=1, stride=1, l2=weight_decay)
    # stem_branch2
    feat_stem2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(feat_stem2)
    # concat
    feat_stem = tf.concat([feat_stem1, feat_stem2], axis=-1)
    return feat_stem


def _bottleneck2(inputs, filters, kernel, s, expand_channel, t, kt=0, idx=0, res=True, l2=0):
    grown_rate = filters//t
    block_name = 'block%d' % idx
    # expand
    x = _conv_block(inputs, expand_channel, name=block_name+'_head', kernel_size=1, l2=l2)
    # if grown_rate > 32:
    #     x = SPPBottleneck(x, expand_channel, expand_channel, kernel_sizes=(3, 7, 9))
    output = []
    for i in range(t):
        if i >= kt:
            kernel = 3
        # dwconv
        dw_name = block_name+'_dw%d' % i
        x2 = _dwconv_block(x, grown_rate, name=dw_name, kernel_size=kernel, pad='same', l2=l2)
        output.append(x2)
        if i < t-1 and expand_channel != grown_rate:
            nl_func = 'relu'
            # if expand_channel // grown_rate >= 4:
            #     nl_func = 'hswish'  # 202.12ms
            x2 = _conv_block(x2, expand_channel, name=block_name+'_exp%d' % i, kernel_size=1, l2=l2, act=nl_func)
            if res:
                x = x + x2
            else:
                x = x2
    return output


def _bottleneck3(inputs, filters, kernel, s, expand_channel, t=2, idx=0, l2=0):
    grown_rate = filters//t
    block_name = 'block%d' % idx
    # expand
    #  with out expand, Train acc: 78.46%; valacc: 78.52%
    x = inputs
    output = []
    for i in range(t):
        # dwconv
        dw_name = block_name+'_dw%d' % i
        x2 = _dwconv_block(x, grown_rate, name=dw_name, kernel_size=kernel, pad='same', l2=l2)
        output.append(x2)
        x = x2
    return output


def create_backbone(inputs, input_shape, weight_decay, num_init_features=32):
    s = input_shape[0] // 4
    feat_x = my_stem_block2(inputs, 32, weight_decay)
    # feat_x = Focus_block(inputs, 32, 3, weight_decay=weight_decay)
    # inverted dense
    block_config = [2, 4, 8, 6]
    growth_rate = [32, 32, 32, 48]
    expand_channel = [32, 64, 128, 192]
    feature_out, blockoutputs = [], []
    total_filter = num_init_features
    len_block = len(block_config)
    for idx, num_layers in enumerate(block_config):
        if idx > 0:
            if idx == 3:
                kt = num_layers
                kernel = 5
            else:
                kt = 0
                kernel = 3
            # res = True if idx > 1 else False
            res = True
            x = _bottleneck2(feat_x, growth_rate[idx]*num_layers, kernel, 1, expand_channel[idx], num_layers, kt, idx, res, l2=weight_decay)
        else:
            x = _bottleneck3(feat_x, growth_rate[idx]*num_layers, 3, 1, expand_channel[idx], num_layers, idx, l2=weight_decay)
        # concat featrue
        feat_x = tf.concat([feat_x]+x, axis=-1)
        total_filter += num_layers * growth_rate[idx]
        blockoutputs.append('{}*{}*{}'.format(s, s, total_filter))
        s = s//2
        feat_x = _conv_block(feat_x, total_filter, kernel_size=1, stride=1, name='Trans%d' % idx, l2=weight_decay)
        # Attention
        feat_x = SE_attention(feat_x, total_filter, 4, weight_decay, hsigmoid=True, name='SE%d' % idx)
        if idx > 0:
            feature_out.append(feat_x)
        # downsample
        if idx < len_block-1:
            feat_x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(feat_x)
            # feat_x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(feat_x)

    return feature_out, blockoutputs


def create_backbone_2(inputs, input_shape, weight_decay, num_init_features=32):
    s = input_shape[0] // 4
    feat_x = my_stem_block2(inputs, 32, weight_decay)
    # inverted dense
    trans_channel = [0, 0, 256, 512]
    block_config = [2, 4, 8, 6]
    growth_rate = [32, 32, 32, 48]
    expand_channel = [32, 64, 128, 192]
    feature_out, blockoutputs = [], []
    total_filter = num_init_features
    len_block = len(block_config)
    for idx, num_layers in enumerate(block_config):
        if idx > 0:
            if idx == 3:
                kt = num_layers
                kernel = 5
            else:
                kt = 0
                kernel = 3
            # res = True if idx > 1 else False
            res = True
            x = _bottleneck2(feat_x, growth_rate[idx]*num_layers, kernel, 1, expand_channel[idx], num_layers, kt, idx, res, l2=weight_decay)
        else:
            x = _bottleneck3(feat_x, growth_rate[idx]*num_layers, 3, 1, expand_channel[idx], num_layers, idx, l2=weight_decay)
        # concat featrue
        feat_x = tf.concat([feat_x]+x, axis=-1)
        if idx > 1:
            total_filter = trans_channel[idx]
        else:
            total_filter += num_layers * growth_rate[idx]
        blockoutputs.append('{}*{}*{}'.format(s, s, total_filter))
        s = s//2
        feat_x = _conv_block(feat_x, total_filter, kernel_size=1, stride=1, name='Trans%d' % idx, l2=weight_decay)
        # Attention
        feat_x = SE_attention(feat_x, total_filter, 4, weight_decay, hsigmoid=True, name='SE%d' % idx)
        if idx > 0:
            feature_out.append(feat_x)
        # downsample
        if idx < len_block-1:
            feat_x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(feat_x)
            # feat_x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(feat_x)

    return feature_out, blockoutputs


def TaskalignedHead(feat_pan, num_channels, num_classes, weight_decay):
    """
     Task-aligned One-stage Object Detection
    """
    bias_init = -np.log((1-.01)/.01)
    det_out = []
    # interactive feats 128*3  ==>  attention (128*3)^2 / 2
    for i, feats in enumerate(feat_pan):
        y = _conv_block(feat_pan[i], num_channels, kernel_size=1, act='relu', l2=weight_decay)
        # se attention ...
        feat_rgn = SE_attention(y, num_channels, 2, weight_decay, hsigmoid=False)
        feat_cls = SE_attention(y, num_channels, 2, weight_decay, hsigmoid=False)
        # Parallel branches ...
        # location
        feat_rgn = _dwconv_block(feat_rgn, num_channels, kernel_size=3, pad='same', act='relu', l2=weight_decay)
        feat_rgn = _dwconv_block(feat_rgn, num_channels, kernel_size=3, pad='same', act='relu', l2=weight_decay)
        # classify
        feat_cls = _dwconv_block(feat_cls, num_channels, kernel_size=3, pad='same', act='relu', l2=weight_decay)
        feat_cls = _dwconv_block(feat_cls, num_channels, kernel_size=3, pad='same', act='relu', l2=weight_decay)
        # attention map
        y = tf.concat([feat_cls, feat_rgn], axis=-1)
        sam = _conv_block(y, 2, kernel_size=5, l2=weight_decay, pad='same', act='sigm', name='TA_Sam%d' % i)
        feat_cls *= sam[..., 0:1]
        feat_rgn *= sam[..., 1:]
        y_loc = layers.Conv2D(4, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                              kernel_initializer=initializers.HeNormal())(feat_rgn)
        y_obj = layers.Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                              kernel_initializer=initializers.HeNormal(),
                              bias_initializer=initializers.Constant(bias_init))(feat_rgn)
        y_cls = layers.Conv2D(num_classes, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                              kernel_initializer=initializers.HeNormal(),
                              bias_initializer=initializers.Constant(bias_init))(feat_cls)
        # out
        det_out.append(tf.concat([y_obj, y_loc, y_cls], -1))
    return det_out


def SPPCSP(x, hidden_ch=256, act='relu', l2=5e-4):
    """
    from YOLOv7.
    """
    cem = _conv_block(x, hidden_ch, kernel_size=1, act=act, l2=l2)
    # branch1
    cem0 = _dwconv_block(cem, hidden_ch//2, kernel_size=5, pad='same', act=act, l2=l2)
    cem1 = layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(cem0)
    cem2 = layers.MaxPool2D(pool_size=(9, 9), strides=(1, 1), padding='same')(cem0)
    cem0 = tf.concat([cem0, cem1, cem2], axis=-1)
    branch1 = _conv_block(cem0, hidden_ch//2, kernel_size=1, act=act, l2=l2)
    branch1 = _dwconv_block(branch1, hidden_ch//2, kernel_size=3, pad='same', act=act, l2=l2)
    # branch2
    branch2 = _conv_block(cem, hidden_ch//2, kernel_size=1, act=act, l2=l2)

    cem = tf.concat([branch1, branch2], axis=-1)
    return cem


def PAFPN_cem2(features, in_channels, out_channels=[128, 256, 512], act='relu', l2=5e-4):
    """PANet"""
    c3, c4, c5 = features
    feat_shape4 = tf.shape(c4)[1:3]
    feat_shape3 = tf.shape(c3)[1:3]
    # SPPCSP
    fpn5 = SPPCSP(c5, l2=l2)
    # top-down path
    fpn5 = _conv_block(fpn5, out_channels[1], kernel_size=1, l2=l2)

    fpn5_upsample = tf.image.resize(fpn5, feat_shape4)
    fpn4 = tf.concat([fpn5_upsample, c4], -1)   # 512->1024/16
    fpn4 = CSP_layer(fpn4, in_channels[1]+out_channels[1], out_channels[1], 3, 2, l2=l2)  # 1024->512/16
    fpn4 = _conv_block(fpn4, out_channels[0], kernel_size=1, l2=l2, act=act)

    c4_upsample = tf.image.resize(fpn4, feat_shape3)
    fpn3 = tf.concat([c4_upsample, c3], -1)   # concat[256, 256] --> 512 /8
    pan_out3 = CSP_layer(fpn3, in_channels[0]+out_channels[0], out_channels[0], 3, 2, l2=l2)
    pan_out3 *= _conv_block(pan_out3, 1, kernel_size=5, pad='same', name='SA0', l2=l2, act='sigm')
    # bottom-up
    fpn3_downsample = _dwconv_block(pan_out3, out_channels[0], kernel_size=2, stride=2,  pad='same', l2=l2)
    pan_out4 = tf.concat([fpn4, fpn3_downsample], -1)
    pan_out4 = CSP_layer(pan_out4, out_channels[1], out_channels[1], 3, 2, l2=l2)
    pan_out4 *= _conv_block(pan_out4, 1, kernel_size=5, pad='same', name='SA1', l2=l2, act='sigm')

    fpn4_downsample = _dwconv_block(pan_out4, out_channels[1], kernel_size=3, stride=2,  pad='same', l2=l2)
    pan_out5 = tf.concat([fpn5, fpn4_downsample], -1)
    pan_out5 = CSP_layer(pan_out5, out_channels[2], out_channels[2], 3, 2, l2=l2)
    pan_out5 *= _conv_block(pan_out5, 1, kernel_size=5, pad='same', name='SA2', l2=l2, act='sigm')

    return [pan_out5, pan_out4, pan_out3, fpn5, fpn4]


def detectLayer(x, num_classes, weight_decay):
    bias_init = -np.log((1-.01)/.01)
    y_loc = layers.Conv2D(4, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                          kernel_initializer=initializers.HeNormal())(x)
    y_obj = layers.Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                          kernel_initializer=initializers.HeNormal(),
                          bias_initializer=initializers.Constant(bias_init))(x)
    y_cls = layers.Conv2D(num_classes, (1, 1), kernel_regularizer=regularizers.l2(weight_decay),
                          kernel_initializer=initializers.HeNormal(),
                          bias_initializer=initializers.Constant(bias_init))(x)
    # out
    return tf.concat([y_obj, y_loc, y_cls], -1)


def create_detectorRDD(input_shape, num_classes, train=False, deepsupervise=False, weight_decay=0.0005):
    # inputs
    if train:
        inputs = layers.Input(shape=(None, None, 3))
    else:
        inputs = layers.Input(shape=(input_shape[0], input_shape[1], 3))

    feature_out, blockoutputs = create_backbone(inputs, input_shape, weight_decay)

    in_channels = [224, 480, 768]
    out_channels = [128, 256, 512]
    # PANet
    feat_pan = PAFPN_cem2(feature_out, in_channels, out_channels, l2=weight_decay, act='relu')

    num_channels = 96
    det_out = TaskalignedHead(feat_pan[:3], num_channels, num_classes, weight_decay)
    if train and deepsupervise:
        det_out.append(detectLayer(feat_pan[3], num_classes, weight_decay))
        det_out.append(detectLayer(feat_pan[4], num_classes, weight_decay))
        det_out.append(detectLayer(feat_pan[2], num_classes, weight_decay))
    print('len of output:', len(det_out))

    model = Model(inputs, det_out)
    return model, blockoutputs


def main(model_name):
    input_shape = (416, 416)
    num_classes = 4
    model, blockoutputs = create_detectorRDD(input_shape, num_classes, train=False, deepsupervise=False)
    # model = Model(inputs, outputs)
    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)
    # save model
    save_model(model, model_name+'.h5')
    model.summary()
    for n in blockoutputs:
        print(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='model name')

    args = parser.parse_args(sys.argv[1:])
    main(args.name)
