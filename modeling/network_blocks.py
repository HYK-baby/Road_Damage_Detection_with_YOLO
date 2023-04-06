import tensorflow as tf
from tensorflow.keras import layers, activations, regularizers, initializers
# from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K


def _dwconv_block(x, num_filters, name=None, kernel_size=3, stride=1, pad='valid', act='relu', l2=0.0005):
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=stride, padding=pad, use_bias=False,
                               kernel_initializer=initializers.HeNormal(),
                               depthwise_regularizer=regularizers.l2(l2), name=name)(x)
    x = layers.BatchNormalization(name=name+'_BN' if name is not None else None)(x)
    # x = layers.BatchNormalization(momentum=.997, epsilon=1e-4, name=name+'_BN' if name is not None else None)(x)
    # if act is not None:
    #     if act == 'relu':
    #         x = layers.ReLU(name=name+'_dw_relu' if name is not None else name)(x)
    if name is not None:
        name += '_pw'
    x = _conv_block(x, num_filters, name, kernel_size=1, l2=l2, act=act)
    # return layers.LeakyReLU(name='conv_pw_%d_relu' % block_id)(x)
    return x


def _dwconv_dilation(x, num_filters, name=None, kernel_size=3, stride=1, pad='valid', dilation=1, act='relu', l2=0.0005):
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=stride, dilation_rate=dilation, padding=pad, use_bias=False,
                               kernel_initializer=initializers.HeNormal(),
                               depthwise_regularizer=regularizers.l2(l2), name=name)(x)
    x = layers.BatchNormalization(momentum=.99, name=name+'_BN' if name is not None else None)(x)
    # if act is not None:
    #     if act == 'relu':
    #         x = layers.ReLU(name=name+'_dw_relu' if name is not None else name)(x)
    if name is not None:
        name += '_pw'
    x = _conv_block(x, num_filters, name, kernel_size=1, l2=l2, act=act)
    # return layers.LeakyReLU(name='conv_pw_%d_relu' % block_id)(x)
    return x


def hard_swish(x):
    """
    Hard swish
    """
    return x * activations.relu(x + 3.0, max_value=6.0) * 0.1666
    # return x * activations.sigmoid(x)


def _conv_block(x, num_filters, name=None, kernel_size=3, stride=1, pad='valid', l2=.0005, act='relu'):
    x = layers.Conv2D(num_filters, (kernel_size, kernel_size), strides=stride, padding=pad, use_bias=False, name=name,
                      kernel_regularizer=regularizers.l2(l2), kernel_initializer=initializers.HeNormal())(x)
    x = layers.BatchNormalization(name=name+'_BN' if name is not None else None)(x)
    # x = layers.BatchNormalization(momentum=.997, epsilon=1e-4, name=name+'_BN' if name is not None else None)(x)
    # activation
    if act is None or act == '':
        return x
    elif act == 'hswish':
        x = hard_swish(x)
    elif act == 'leaky':
        x = layers.LeakyReLU(alpha=0.2)(x)
    elif act == 'sigm':
        x = activations.sigmoid(x)
    elif act == 'silu':
        x = x * activations.sigmoid(x)
    elif act == 'mish':
        x = x * tf.math.tanh(tf.math.softplus(x))
    else:
        x = activations.relu(x)
    return x


def SPPBottleneck(x, in_channels, out_channels, kernel_sizes=(5, 9, 13), act='silu', weight_decay=5e-4):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    mid_channels = in_channels // 2
    x = _conv_block(x, mid_channels, kernel_size=1, name='spp_conv1', act=act, l2=weight_decay)
    # tf.while()
    feat_list = [layers.MaxPool2D(pool_size=(ks, ks), strides=(1, 1), padding='same')(x) for ks in kernel_sizes]
    x = tf.concat([x] + feat_list, axis=-1)
    x = _conv_block(x, out_channels, kernel_size=1, name='spp_conv2', act=act, l2=weight_decay)
    return x


def _bottleneck_res(x, out_channels, expansion=0.5, shortcut=True, depthwise=False, act='silu', weight_decay=5e-4):
    mid_channels = int(out_channels * expansion)  # middle channels
    y = _conv_block(x, mid_channels, kernel_size=1, act=act, l2=weight_decay)
    if depthwise:
        y = _dwconv_block(y, out_channels, kernel_size=3, pad='same', act=act, l2=weight_decay)
    else:
        y = _conv_block(y, out_channels, kernel_size=3, pad='same', act=act, l2=weight_decay)
    if shortcut:
        y = y + x
    return y


def CSP_resblock(x, out_channels, depth, shortcut=True, stage_name=None, expansion=0.5, depthwise=False, act='silu', weight_decay=5e-4):
    # stage_name = 'stage{}'.format(blockid)
    mid_channels = int(out_channels * expansion)  # middle channels
    # x1 = _conv_block(x, mid_channels, name=stage_name+'_p1' if stage_name is not None else None, kernel_size=1, act=act)
    # x2 = _conv_block(x, mid_channels, name=stage_name+'_p2' if stage_name is not None else None, kernel_size=1, act=act)
    x = _conv_block(x, out_channels, name=stage_name+'_p' if stage_name is not None else None, kernel_size=1, act=act, l2=weight_decay)
    x1 = x[..., :mid_channels]
    x2 = x[..., mid_channels:]
    for i in range(depth):
        x1 = _bottleneck_res(x1, mid_channels, 1, shortcut, depthwise, act)

    x = tf.concat([x1, x2], axis=-1)
    x = _conv_block(x, out_channels, name=stage_name+'trans' if stage_name is not None else None, kernel_size=1, act=act, l2=weight_decay)
    return x


def GlobalContextBlock(x, channels, ratio=4, block_id=1, l2=5e-4):
    # W_k
    mask = layers.Conv2D(1, (1, 1), name='gc%d_conv_k' % block_id,
                         kernel_regularizer=regularizers.l2(l2), kernel_initializer=initializers.HeNormal())(x)
    # [N, h, w, 1]
    # mask = tf.nn.softmax(mask)
    mask = activations.sigmoid(mask)
    # context: [N, 1, 1, C]
    context = mask * x
    context = tf.reduce_sum(context, axis=[1, 2], keepdims=True)
    # transform W_v
    context = _conv_block(context, channels//ratio, name='gc%d_conv_v1' % block_id, kernel_size=1, l2=l2)
    context = _conv_block(context, channels, name='gc%d_conv_v2' % block_id, kernel_size=1, l2=l2, act='')
    # fusion
    out = x + context
    return out


def SE_attention(x, channels, r=4, l2=5e-4, hsigmoid=False, name=None):
    """
    sequence and extract.
    """
    excitation_name1 = None if name is None else name + '1'
    excitation_BN1 = None if name is None else name + '_BN1'
    excitation_name2 = None if name is None else name + '2'
    excitation_BN2 = None if name is None else name + '_BN2'
    squeeze = tf.reduce_mean(x, [1, 2], keepdims=True)
    excitation = layers.Conv2D(channels//r, 1, use_bias=False, name=excitation_name1,
                               kernel_initializer=initializers.HeNormal(),
                               kernel_regularizer=regularizers.l2(l2))(squeeze)
    excitation = layers.BatchNormalization(momentum=.99, name=excitation_BN1)(excitation)
    excitation = activations.relu(excitation)
    excitation = layers.Conv2D(channels, 1, use_bias=False, name=excitation_name2,
                               kernel_initializer=initializers.HeNormal(),
                               kernel_regularizer=regularizers.l2(l2))(excitation)
    excitation = layers.BatchNormalization(momentum=.99, name=excitation_BN2)(excitation)
    if hsigmoid:
        excitation = activations.relu(excitation + 3.0, max_value=6.0) * .16666
    else:
        excitation = activations.sigmoid(excitation)  # shape=(b, c)
    x = excitation * x
    return x


def Focus_block(inputs, out_channels, ksize=3, stride=1, weight_decay=5e-4):
    """Focus width and height information into channel space."""
    # shape of x (b,w,h,c) -> y(b,w/2,h/2,4c)
    patch_top_left = inputs[::, ::2, ::2]
    patch_top_right = inputs[::, ::2, 1::2]
    patch_bot_left = inputs[::, 1::2, ::2]
    patch_bot_right = inputs[::, 1::2, 1::2]
    x = tf.concat([patch_top_left, patch_top_right, patch_bot_left, patch_bot_right], axis=-1)
    x = _conv_block(x, out_channels, name=None, kernel_size=ksize, stride=stride, pad='same', act='silu', l2=weight_decay)
    return x


def PAFPN(features, in_channels, out_channels=[128, 256, 512], act='relu', l2=5e-4):
    """PANet"""
    c3, c4, c5 = features
    feat_shape4 = tf.shape(c4)[1:3]
    feat_shape3 = tf.shape(c3)[1:3]
    # CEM. with CEM 383ms
    # cem = _dwconv_block(c5[..., :in_channels[2]//2], 256, kernel_size=3, pad='same', l2=l2)
    # cem += _dwconv_block(cem, 256, kernel_size=3, pad='same', l2=l2)
    # cem += _dwconv_block(cem, 256, kernel_size=3, pad='same', l2=l2)
    # fpn5 = tf.concat([cem, c5[..., in_channels[2]//2:]], axis=-1)
    # CEM2.
    cem = _conv_block(c5, 256, kernel_size=1, l2=l2)
    cem1 = _dwconv_dilation(cem, 128, kernel_size=5, pad='same', dilation=1, l2=l2)
    cem2 = _dwconv_dilation(cem, 128, kernel_size=5, pad='same', dilation=2, l2=l2)
    fpn5 = tf.concat([cem, cem1, cem2], axis=-1)
    # CEM3.
    # cem = _conv_block(c5, 256, kernel_size=1, l2=l2)
    # cem1 = _dwconv_dilation(cem, 128, kernel_size=5, pad='same', dilation=1, l2=l2)
    # cem2 = _dwconv_dilation(cem, 128, kernel_size=5, pad='same', dilation=2, l2=l2)
    # cem3 = _dwconv_dilation(cem, 128, kernel_size=5, pad='same', dilation=3, l2=l2)
    # fpn5 = tf.concat([cem, cem1, cem2, cem3], axis=-1)
    # top-down path
    fpn5 = _conv_block(fpn5, out_channels[1], kernel_size=1, l2=l2)

    fpn5_upsample = tf.image.resize(fpn5, feat_shape4)
    fpn4 = tf.concat([fpn5_upsample, c4], -1)   # 512->1024/16
    fpn4 = CSP_layer(fpn4, 2*out_channels[1], out_channels[1], 3, 2, l2=l2)  # 1024->512/16
    fpn4 = _conv_block(fpn4, out_channels[0], kernel_size=1, l2=l2, act=act)

    c4_upsample = tf.image.resize(fpn4, feat_shape3)
    fpn3 = tf.concat([c4_upsample, c3], -1)   # concat[256, 256] --> 512 /8
    pan_out3 = CSP_layer(fpn3, 2*out_channels[0], out_channels[0], 3, 2, l2=l2)  # 512->256/8
    pan_out3 *= _conv_block(pan_out3, 1, kernel_size=5, pad='same', name='SA0', l2=l2, act='sigm')
    # bottom-up
    fpn3_downsample = _dwconv_block(pan_out3, out_channels[0], kernel_size=2, stride=2,  pad='same', l2=l2)
    pan_out4 = tf.concat([fpn4, fpn3_downsample], -1)
    pan_out4 = CSP_layer(pan_out4, 2*out_channels[0], out_channels[1], 3, 2, l2=l2)
    pan_out4 *= _conv_block(pan_out4, 1, kernel_size=5, pad='same', name='SA1', l2=l2, act='sigm')

    fpn4_downsample = _dwconv_block(pan_out4, out_channels[1], kernel_size=3, stride=2,  pad='same', l2=l2)
    pan_out5 = tf.concat([fpn5, fpn4_downsample], -1)
    pan_out5 = CSP_layer(pan_out5, 2*out_channels[1], out_channels[2], 3, 2, l2=l2)
    pan_out5 *= _conv_block(pan_out5, 1, kernel_size=5, pad='same', name='SA2', l2=l2, act='sigm')

    return [pan_out5, pan_out4, pan_out3]


def CSP_layer(x, in_ch, out_ch, k, depth, act='relu', exp=.5, l2=5e-4):
    h_ch1 = int(exp*out_ch)
    h_ch2 = out_ch - h_ch1
    x = _conv_block(x, out_ch, kernel_size=1, act=act, l2=l2)
    x1 = x[..., :h_ch1]
    x2 = x[..., h_ch1:]
    # x1 = _conv_block(x, h_ch1, kernel_size=1, l2=l2)
    # x2 = _conv_block(x, h_ch2, kernel_size=1, l2=l2)
    for i in range(depth):
        x2 = _dwconv_block(x2, h_ch2, kernel_size=k, pad='same', act=act, l2=l2)
    x = tf.concat([x1, x2], -1)
    # x = _conv_block(x, out_ch, kernel_size=1, l2=l2)
    return x


def CSP_Tree_layer(x, out_ch, exp=.5, act='relu', l2=5e-4):
    # hswish
    h_ch1 = int(exp*out_ch)
    x = _conv_block(x, out_ch, kernel_size=1, act=act, l2=l2)
    x0 = x[..., :h_ch1]
    x3 = x[..., h_ch1:]
    # branch1
    x1 = _dwconv_block(x0, h_ch1//2, kernel_size=3, pad='same', act=act, l2=l2)
    shared_x = _conv_block(x1, h_ch1, kernel_size=1, act=act, l2=l2)
    shared_x += x0
    x2 = _dwconv_dilation(shared_x, h_ch1//2, kernel_size=3, pad='same', dilation=2, act=act, l2=l2)
    # branch2
    x = tf.concat([x1, x2, x3], axis=-1)
    x = SE_attention(x, out_ch, 4, l2=l2, hsigmoid=True)
    return x


def CSP_layer2(x, in_ch, out_ch, k, depth, exp=.5, l2=5e-4):
    h_ch1 = int(exp*out_ch)
    h_ch2 = out_ch - h_ch1
    x2 = x[..., :h_ch1]
    x1 = x[..., h_ch1:]
    for i in range(depth):
        x2 = _dwconv_block(x2, h_ch2, kernel_size=k, pad='same', l2=l2)
    x = tf.concat([x1, x2], -1)
    # x = _conv_block(x, out_ch, kernel_size=1, l2=l2)
    return x


def classify_head(feature, num_classes):
    x = layers.GlobalAveragePooling2D()(feature)
    y = layers.Dense(num_classes, activation='softmax')(x)
    return y
