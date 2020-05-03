# coding=utf-8
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
picture_main_dir = '../custom_training_data/'
# picture_main_dir = '../custom_training_data_edge/'

filename_extension= '_resnext_normal_tembusu'

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
import utilModel


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
# model = utilModel.from_paper(input_shape,nclass)
#model = utilModel.VGG19(input_shape,nclass)
# model = utilModel.CaffeNet(input_shape,nclass)
model = utilModel.ResNeXt50(input_shape=input_shape, nclass=nclass)
# model = utilModel.InceptionResNet2(input_shape=input_shape, nclass=nclass)

import time
start_time = time.time()

history = model.fit_generator(train_gen,
                              steps_per_epoch=nclass*1000/batch_size,          
                              validation_data=val_gen,
                              validation_steps=batch_size, 
                              epochs=40, verbose=1,
                              callbacks=callbacks_list)

end_time = time.time()

f = open("./output/time" + filename_extension + ".txt","w+")
f.write("time for " + filename_extension + " is %d seconds" % (end_time-start_time))
f.close()

model.save('model' + filename_extension + '.h5')


"""
VISUALISATION
================================================================================
"""
import utilVisualisation

utilVisualisation.visualize(history, filename_extension)