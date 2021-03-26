
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
import tensorflow.keras.backend as K
import tensorflow as tf
from config import patience, epochs, num_train_samples, num_lfw_valid_samples, batch_size,alpha
from data_generator import DataGenSequence
from model import build_model

# 定义triplet loss损失函数
def triplet_loss(y_ture,y_pred):
    batch_num = len(y_pred)
    a_pred = y_pred[0:int(batch_num/3)]
    p_pred = y_pred[int(batch_num/3):int(2*batch_num/3)]
    n_pred = y_pred[int(2*batch_num/3):batch_num]
    positive_distance = K.square(tf.norm(a_pred - p_pred, axis=-1))
    negative_distance = K.square(tf.norm(a_pred - n_pred, axis=-1))
    loss = K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))
    return loss

if __name__ == '__main__':

    # 模型保存路径
    checkpoint_models_path = 'weights/'

    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.h5'
    # 定义模型保存信息
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    # 训练早停
    early_stop = EarlyStopping('val_loss', patience=patience)
    # 学习率衰减策略
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=int(patience / 2), verbose=1)

    log = TensorBoard(log_dir='log')

    # 构建网络模型
    model = build_model()

    # 定义优化器和学习率
    # sgd = keras.optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-6)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    # 编译，使用triplet loss损失函数
    model.compile(optimizer=adam, loss=triplet_loss)

    print(model.summary())

    # 声明回调函数
    callbacks = [model_checkpoint, early_stop, reduce_lr,log]

    # 开始训练
    model.fit_generator(DataGenSequence('face_dataset'),
            steps_per_epoch=num_train_samples // batch_size,
            validation_data=DataGenSequence('face_dataset'),
            validation_steps=num_lfw_valid_samples // batch_size,
            epochs=epochs,verbose=1,callbacks=callbacks,
            use_multiprocessing=True,workers=0)

