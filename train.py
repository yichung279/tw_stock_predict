import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Lambda, Dense,TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
set_session(tf.Session(config = config))

def build_model():
    model = Sequential()
    
    model.add(LSTM(64, input_shape = (20, (5)),  return_sequences = True))
    model.add(LSTM(32,  return_sequences = True))
    model.add(TimeDistributed(Dense(5)))
    model.add(Lambda(lambda x: x[: ,-5:,]))
    model.summary()

    return model
if __name__ == "__main__":
    data = np.load('training_data/all_data.npy')
    x, y = np.split(data, [15], axis=1)

    zeros = np.zeros((x.shape[0], 5, 5))
    x = np.concatenate((x, zeros), axis=1)    

    # print(x.shape, y.shape)
    model = build_model()
   
    modelname = 'LSTM_64_32-500'

    model_ckpt = ModelCheckpoint('model/%s.h5' % modelname, verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./logs/%s' % modelname, histogram_freq=0, write_graph=True, write_images=False)

    model.compile(loss = 'mse', optimizer = Adam(lr = 1e-4))
    model.fit(x = x, y = y, batch_size= 64, epochs = 500, callbacks = [model_ckpt, tensorboard], validation_split=0.4,  shuffle=True)
