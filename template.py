import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'weight.h5'

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

nrows = 100
ncolumns = 100
batch_size = 50


train_gen = train_datagen.flow_from_directory(directory= picture_main_dir + 'train/',
                                              target_size=(100, 100),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=0)
val_gen = val_datagen.flow_from_directory(directory= picture_main_dir + 'val/',
                                          target_size=(100, 100),
                                          color_mode='rgb',
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False,
                                          seed=0)

test_gen = test_datagen.flow_from_directory(directory= picture_main_dir + 'test/',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False,
                                            seed=0)

"""
MODEL
====================================================================================
"""



"""
CaffeNet
"""
def CaffeNet(input_shape=(100,100,1),nclass=24):
    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='same'))
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
    model.add(Dense(nclass))
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model


"""
VGG19
====================================================================================
"""

def VGG19(input_shape=(100,100,1),nclass=24): 
    model = Sequential()

    # Block 1
    model.add(Conv2D(filters=64,
                     input_shape=input_shape,
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
    model.add(Dense(nclass, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model


"""
from_paper
====================================================================================
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
                  optimizer='adam', metrics=['accuracy'])
    
    return model

"""
====================================================================================
"""



"""
TRAINING
==================================================================================
Change filepath accordingly
"""


checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                        verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                      mode='min', 
                      patience=4, restore_best_weights=True)

callbacks_list = [checkpoint, early]


input_shape=(nrows,ncolumns,1)
nclass=24
model = from_paper(input_shape,nclass)
#model = VGG19(input_shape,nclass)
#model = CaffeNet(input_shape,nclass)
history = model.fit_generator(train_gen,
                              steps_per_epoch=nclass*1000/batch_size,          
                              validation_data=val_gen,
                              validation_steps=batch_size, 
                              epochs=40, verbose=1,
                              callbacks=callbacks_list)

model.save_model('model.h5')


"""
VISUALISATION
================================================================================
"""


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

plt.savefig('accuracy_loss.png')