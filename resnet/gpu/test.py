from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers, activations
import os
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import math

train_datagen =  ImageDataGenerator(rescale=1.0/255, 
                                    rotation_range=10,
                                    zoom_range=0.1)

val_datagen =  ImageDataGenerator(rescale=1.0/255)

test_datagen =  ImageDataGenerator(rescale=1.0/255)



train_gen = train_datagen.flow_from_directory(directory='./train_copy/train/',
                                              target_size=(100, 100),
                                              color_mode='grayscale',
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=0)
                                              
val_gen = val_datagen.flow_from_directory(directory='./train_copy/val/',
                                          target_size=(100, 100),
                                          color_mode='grayscale',
                                          batch_size=32,
                                          class_mode='categorical',
                                          shuffle=False,
                                          seed=0)

test_gen = test_datagen.flow_from_directory(directory='./train_copy/test/',
                                            target_size=(100, 100),
                                            color_mode='grayscale',
                                            batch_size=32,
                                            class_mode=None,
                                            shuffle=False,
                                            seed=0)



cnn = Sequential()

cnn.add(Conv2D(64,kernel_size=(3,3),
               activation='relu',
               input_shape=(100,100,1),
               padding='same'))
cnn.add(Conv2D(64,kernel_size=(3,3),
               activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.5))

cnn.add(Conv2D(128,kernel_size=(3,3),
               activation='relu'))
cnn.add(Conv2D(128,kernel_size=(3,3),
               activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.5))

cnn.add(Flatten())

cnn.add(Dense(512,
              activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(512,
              activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(29,activation='softmax'))

cnn.compile(Adam(lr=0.0001), 
            loss='categorical_crossentropy',
            metrics=['accuracy'])


filepath = 'tes_weight_gpu_100.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                      mode='min', 
                      patience=4)
callbacks_list = [checkpoint, early]


history = cnn.fit_generator(train_gen,
                            steps_per_epoch=math.ceil((29.0 * 3000)/32),
                            validation_data=val_gen,
                            validation_steps=32, 
                            epochs=20, verbose=1,
                            callbacks=callbacks_list)

