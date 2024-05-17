import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(InputLayer(input_shape=(784,)))
    model.add(Dense(512, 'relu'))
    model.add(Dropout(0.2))
    # Hidden Layer
    model.add(Dense(512, 'relu'))
    model.add(Dropout(0.2))
    # Output Layer
    model.add(Dense(10, 'softmax'))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr=0.03):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)