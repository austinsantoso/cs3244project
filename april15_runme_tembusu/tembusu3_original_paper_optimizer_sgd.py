# coding=utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
train_datagen =  ImageDataGenerator(rescale=1.0/255, 
                                    rotation_range=15,
                                    zoom_range=0.2)

val_datagen =  ImageDataGenerator(rescale=1.0/255)

test_datagen =  ImageDataGenerator(rescale=1.0/255)

"""
CHOOSE THE DIRECTORY OF IMAGE
"""
# picture_main_dir = '../train_copy/'
# picture_main_dir = '../custom_training_data/'
picture_main_dir = '../custom_training_data_edge/'

filename_extension= '_original_from_paper__optimizer_sgd'

nrows = 100
ncolumns = 100
batch_size = 50


train_gen = train_datagen.flow_from_directory(directory= picture_main_dir + 'train/',
                                              target_size=(nrows, ncolumns),
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=0)

val_gen = val_datagen.flow_from_directory(directory= picture_main_dir + 'val/',
                                          target_size=(nrows, ncolumns),
                                          color_mode='grayscale',
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False,
                                          seed=0)

test_gen = test_datagen.flow_from_directory(directory= picture_main_dir + 'test/',
                                            target_size=(nrows, ncolumns),
                                            color_mode='grayscale',
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False,
                                            seed=0)

"""
MODEL
====================================================================================
Done in utilMode.py
"""
def from_paper(input_shape=(100,100,1),nclass=24):
    model = Sequential()
    
    model.add(Conv2D(filters=16,
                     input_shape=input_shape,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(5, 5)))
    
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(5, 5)))
    
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dense(nclass, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])
    
    return model


"""
TRAINING
==================================================================================
Change filepath accordingly
"""

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'weight' + filename_extension + '.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                        verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                      mode='min', 
                      patience=40, restore_best_weights=True)

callbacks_list = [checkpoint, early]

"""
Decalre input size and number of classes
"""
input_shape=(nrows,ncolumns,1)
nclass=24

"""
SELECT YOUR MODEL
"""
model = from_paper(input_shape,nclass)


history = model.fit_generator(train_gen,
                              steps_per_epoch=nclass*1000/batch_size,          
                              validation_data=val_gen,
                              validation_steps=batch_size, 
                              epochs=40, verbose=1,
                              callbacks=callbacks_list)

model.save('model' + filename_extension + '.h5')


"""
VISUALISATION
================================================================================
"""
import utilVisualisation

utilVisualisation.visualize(history, filename_extension)