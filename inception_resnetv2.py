from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend

def conv2d_bn(x,filters,kernel_size,strides=1,padding='same',
              activation='relu',use_bias=False,name=None):

    x = layers.Conv2D(filters,kernel_size,strides=strides,
                      padding=padding,use_bias=use_bias,name=name)(x)
    if not use_bias:
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=3,scale=False,name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    mixed = layers.Concatenate(axis=3, name=block_name + '_mixed')(branches)

    up = conv2d_bn(mixed,backend.int_shape(x)[3],1,use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x

def InceptionResNetV2(shape=(160,160,3),pooling=None):
    inputs = layers.Input(shape=shape)
    # Stem block: h x w x 192
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): h x w x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]

    x = layers.Concatenate(axis=3, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): h x w x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,scale=0.17,
                                   block_type='block35',block_idx=block_idx)

    # Mixed 6a (Reduction-A block): h//2 x w//2 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=3, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): h//2 x w//2 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,scale=0.1,
                                   block_type='block17',block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=3, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): h//4 x w//4 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,scale=0.2,
                                   block_type='block8',block_idx=block_idx)
    x = inception_resnet_block(x,scale=1.,block_type='block8',block_idx=10)

    # Final convolution block: h//2 x w//2 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    model = Model(inputs, x, name='inception_resnet_v2')
    return model

if __name__ == "__main__":
    inception_resnet = InceptionResNetV2()
    inception_resnet.summary()