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

#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(100,100,3), kernel_size=(11,11), strides=(4,4), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(29))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'weight_ANCN.h5'
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

