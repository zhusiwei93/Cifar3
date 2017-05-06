import os
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from loaddata import load_cifar3, load_cifar3_test
from predict import save_predict

def save_model(model, extra_name):

    #save model
    model_json = model.to_json()
    with open("model"+extra_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model"+extra_name+".h5")
    print("Saved model to disk: " + extra_name)


# Load CIFAR3 Dataset
(train_features, train_labels), (val_features, val_labels) = load_cifar3()
(test_features, test_sample_labels) = load_cifar3_test()

num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))

# Data pre-processing
train_features = train_features.astype('float32')/255
val_features = val_features.astype('float32')/255

# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
val_labels = np_utils.to_categorical(val_labels, num_classes)

# Convolutional Neural Network for CIFAR-3 dataset

# Define the model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# train the model
start = time.time()

epoch_step = 20
epoch_max  = 200
epoch_count = 0
while epoch_count<=epoch_max:
    # Train the model
    model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 32),
                                     samples_per_epoch = train_features.shape[0], nb_epoch = epoch_step,
                                     validation_data = (val_features, val_labels), verbose=2)
    epoch_count += epoch_step
    extra_name = '-' + str(epoch_count)
    save_predict(test_features, test_sample_labels, model, extra_name, datagen)
    save_model(model, extra_name)
end = time.time()

save_model(model,"")
print "Model took %0.2f seconds to train"%(end - start)
