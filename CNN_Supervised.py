#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN supervised
"""
from six.moves import cPickle as pickle
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

"""
Opening pickled datasets
"""

pfile = r"./Data/WaveImgDatasets.pickle"
with (open(pfile, "rb")) as openfile:
    while True:
        try:
            WIData = pickle.load(openfile)
        except EOFError:
            break

X_test = WIData["test_datasets"]
Y_test = WIData["test_labels"]
X_train = WIData["train_datasets"]
Y_train = WIData["train_labels"]

"""
one hot-encoding
validation dataset
"""
# one-hot encode the labels
num_classes = 10
Y_train_hot = keras.utils.to_categorical(Y_train-1, num_classes)
Y_test_hot = keras.utils.to_categorical(Y_test-1, num_classes)

# break training set into training and validation sets
(X_train, X_valid) = X_train[2000:], X_train[:2000]
(Y_train, Y_valid) = Y_train_hot[2000:], Y_train_hot[:2000]
Y_test = Y_test_hot

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')

"""
CNN modeling
1 Channel
"""
CNNch = 2

model = Sequential()
#1
model.add(Conv2D(filters=32, kernel_size=(5,5),strides = (1,1), padding='same', activation='relu', 
                        input_shape=(45,45,CNNch)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#2
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides = (1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
#3
model.add(Conv2D(filters=128, kernel_size=(5, 5), strides = (1, 1), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

"""
CNN training
20 epochs, test/train/validation accuracy 100%
without learning, 10%.
"""
# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = model.fit(X_train[:,:,:,0:CNNch], Y_train, batch_size=32, epochs=10,
          validation_data=(X_valid[:,:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

"""
Evaluating accuracty on test
"""
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# evaluate and print test accuracy
score = model.evaluate(X_test[:,:,:,:CNNch], Y_test, verbose=0)
print('\n', 'CNN Test accuracy:', score[1])

score = model.evaluate(X_train[:,:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'CNN train accuracy:', score[1])

score = model.evaluate(X_valid[:,:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'CNN validation accuracy:', score[1])
