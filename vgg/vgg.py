from keras.preprocessing.image import ImageDataGenerator
train_datagen =  ImageDataGenerator(rescale=1.0/255, 
                                    rotation_range=10,
                                    zoom_range=0.2)

val_datagen =  ImageDataGenerator(rescale=1.0/255)

test_datagen =  ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(directory='../train_copy/train/',
                                              target_size=(100, 100),
                                              color_mode='rgb',
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=0)
val_gen = val_datagen.flow_from_directory(directory='../train_copy/val/',
                                          target_size=(100, 100),
                                          color_mode='rgb',
                                          batch_size=32,
                                          class_mode='categorical',
                                          shuffle=False,
                                          seed=0)

test_gen = test_datagen.flow_from_directory(directory='../train_copy/test/',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode=None,
                                            shuffle=False,
                                            seed=0)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()

# Block 1
model.add(Conv2D(filters=64,
                 input_shape=(100,100,3),
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
          
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))

# Block 2
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))
          
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))

# Block 3
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))

# Block 4
model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))

# Block 5
model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))

# Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, activation='relu'))

# 2nd Fully Connected Layer
model.add(Dense(4096, activation='relu'))

# Output layer
model.add(Dense(29, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'weight_VGG19.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                        verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                      mode='min', 
                      patience=4)
callbacks_list = [checkpoint, early]

history = model.fit_generator(train_gen,
                              steps_per_epoch=29*3000/32,          
                              validation_data=val_gen,
                              validation_steps=32, 
                              epochs=20, verbose=1,
                              callbacks=callbacks_list)