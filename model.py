from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Lambda,concatenate
from tensorflow.keras import backend as K
from inception_resnetv2 import InceptionResNetV2

# 构建网络模型函数
def build_model(input_shape=(160, 160, 3),classes=128):

    base_model = InceptionResNetV2(shape=input_shape,pooling='avg')
    image_input = base_model.input
    image_features = base_model.layers[-1].output
    image_vector = Dense(classes)(image_features)
    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
    output = normalize(image_vector)

    model = Model(inputs=image_input,
                  outputs=output)
    return model

if __name__ == "__main__":
    model,image_embedder = build_model()
    model.summary()