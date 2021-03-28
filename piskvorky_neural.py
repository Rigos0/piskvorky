""""" Slouzi k trainingu policy funkce z daneho datasetu"""""

# armada knihoven na machine learning
import numpy as np
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import tensorflow as tf

# reseni nejakeho bugu v Kerasu...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

""""" Nacti dataset """""
X_train = np.load("positons_policies1.npy")
X_train = tf.expand_dims(X_train, axis=-1)
print(X_train.shape)

y_train = np.load("policies1.npy")
y_train = to_categorical(y_train, num_classes=225)
print(y_train)


"""""""""""""""""""""
Network architecture
"""""""""""""""""""""
LAYER1_SIZE = 16
LAYER2_SIZE = 2

L2_REGULARISATION = 0.0002

model = Sequential()
model.add(Conv2D(128, activation="relu", kernel_size=(5, 5),
                 input_shape=(15, 15, 1),
                 data_format="channels_first",
                 kernel_regularizer=l2(L2_REGULARISATION),
                 padding='same'))


model.add(Conv2D(128, activation="relu", kernel_size=(3, 3),
                 data_format="channels_first",
                 kernel_regularizer=l2(L2_REGULARISATION),
                 padding='same'))
model.add(Dropout(0.3))


model.add(Flatten())

model.add(Dense(500, activation='relu', kernel_regularizer=l2(L2_REGULARISATION)))
model.add(Dense(225, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

"""""""""""""""
Model training
"""""""""""""""

# start the training
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1, shuffle=True,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# save the model in a file
model.save('policy_model')
