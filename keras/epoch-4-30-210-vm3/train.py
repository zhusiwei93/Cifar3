# Import Modules

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

import time
import numpy as np

# % matplotlib inline
np.random.seed(10601)
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD

# Load CIFAR3 Dataset

from loaddata import load_cifar3_train
(train_features, train_labels) = load_cifar3_train()

num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))

# Data pre-processing

train_features = train_features.astype('float32')/255

# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)

def save_model(model, extra_name):

    #save model
    model_json = model.to_json()
    with open("model"+extra_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model"+extra_name+".h5")
    print("Saved model to disk: " + extra_name)

# Convolutional Neural Network for CIFAR-3 dataset

# Define the model

import os.path
model_filepath = 'model.json'
if os.path.exists(model_filepath):

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("[NOTICE] Loaded model from disk")

else:
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same',
                 input_shape=train_features.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
	# initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	# Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zca_whitening=True,		# apply ZCA whitening
							 rotation_range=10,			# randomly rotate images in the range (degrees, 0 to 180)
							 width_shift_range=0.1,		# randomly shift images horizontally (fraction of total width)
							 height_shift_range=0.1,	# randomly shift images vertically (fraction of total height)
							 vertical_flip=False,
                             horizontal_flip=True)

# train the model
start = time.time()

epoch_step = 2
epoch_max  = 210
epoch_count = 0
while epoch_count<=epoch_max:
    # Train the model
    model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                     samples_per_epoch = train_features.shape[0], nb_epoch = epoch_step,
                                     verbose=2)
    epoch_count += epoch_step
    save_model(model, '-' + str(epoch_count))
end = time.time()

save_model(model,"")
print "Model took %0.2f seconds to train"%(end - start)

