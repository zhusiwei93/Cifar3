import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

import time
import numpy as np

np.random.seed(10601)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD

# Load CIFAR3 Dataset

from loaddata import load_cifar3
(train_features, train_labels), (test_features, test_labels) = load_cifar3()

num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))

# Data pre-processing

train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255

# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

# Function to plot model accuracy and loss

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

# Funtion to compute test accuracy

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

# Convolutional Neural Network for CIFAR-10 dataset

# Define the model
model = Sequential()
model.add(Convolution2D(96, (3, 3), activation='relu', border_mode = 'same', input_shape=(3, 32, 32)))
model.add(Convolution2D(96, (3, 3), activation='relu', border_mode='same'))
model.add(Convolution2D(96, 3, 3, border_mode='same', subsample = (2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, (3, 3), activation='relu', border_mode = 'same'))
model.add(Convolution2D(192, (3, 3), activation='relu', border_mode = 'same'))
model.add(Convolution2D(192, (3, 3), border_mode='same', subsample = (2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, (3, 3), activation='relu', border_mode = 'same'))
model.add(Convolution2D(192, (1, 1),activation='relu', border_mode='valid'))
model.add(Convolution2D(3, (1, 1), activation='relu', border_mode='valid'))



model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zca_whitening=True,  		# apply ZCA whitening
							 rotation_range=10,  		# randomly rotate images in the range (degrees, 0 to 180)
							 width_shift_range=0.1,  	# randomly shift images horizontally (fraction of total width)
							 height_shift_range=0.1,  	# randomly shift images vertically (fraction of total height)
							 vertical_flip=False,
                             horizontal_flip=True)


start = time.time()

if True:
	model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
									 samples_per_epoch = train_features.shape[0], nb_epoch = 120,
									 validation_data = (test_features, test_labels), verbose=2)

else:
	model_info = model.fit(train_features, train_labels,
						   batch_size=128, epochs=1,
						   validation_data = (test_features, test_labels),
						   verbose=2)

end = time.time()
score = model.evaluate(test_features, test_labels, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

print "Model took %0.2f seconds to train"%(end - start)

# compute test accuracy
print "Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model)
