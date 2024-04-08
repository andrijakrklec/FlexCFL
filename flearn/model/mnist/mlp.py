import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(InputLayer(input_shape=(784,)))
    # Hidden Layer
    model.add(Dense(128, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # Output Layer
    model.add(Dense(10, 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr=0.03):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)