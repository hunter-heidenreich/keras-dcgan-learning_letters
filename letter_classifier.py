#!/usr/bin/env python3

'''
    001 - batch_size = 16
    002 - batch_size = 32
    003 - batch_size = 8
    004 - epochs = 20, 1st Dropout(0.30)
    005 - 1st Dropout(0.35)                         | Up
    006 - 1st Dropout(0.30)                         | Down  (62.30 | 84.70)
    007 - 1st Dropout(0.40)                         | Down  (62.24 | 84.47)
    008 - 1st Dropout(0.35)                         | Up    (63.02 | 86.53)
    009 - 2nd Dropout(0.4) [from 0.5]               | Up    (69.34 | 86.42)
    010 - 2nd Dropout(0.35)                         | Up    (69.86 | 87.90)
    011 - Creating more data                        | Up    (76.32 | 87.58) {CRASHED}
    012 - epochs = 15, 1st Dense(256) [from 128]    | Up    (80.35 | 89.53)
    013 - 1st Dense(512)                            | Up    (84.73 | 91.60) {CRASHED}
    014 - Duplicate the conv/poolin 64              | Up    (87.16 | 92.98) {CRASHED}
    015 - 3rd Conv2D(128, (3, 3))                   | Up    (88.77 | 94.32)
    016 - 1st Dense(1024)                           | Up    (90.27 | 95.13)
    017 - Duplicate last Dense                      | Down  (90.80 | 94.60)
    018 - Change activiation (LeakyReLU)            | Down  (87.86 | 93.14) {CRASHED}
    019 - Adam Optimizer instead of adadelta        | Up    (91.98 | 97.00)
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt

batch_size = 8
num_classes = 26
epochs = 15

nb_train_samples = 10426
nb_validation_samples = 2470

# input image dimensions
img_rows, img_cols = 32, 32

train_data_dir = 'data/train'
validation_data_dir = 'data/validate'


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

          '''
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.gca().set_ylim([0.0, 1.0])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''
