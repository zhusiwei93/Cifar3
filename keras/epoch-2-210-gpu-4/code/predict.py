import numpy as np
import keras
import os
from loaddata import load_cifar3_test
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from IPython.core.debugger import Tracer
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def save_predict(test_features, test_sample_labels, model, extra_name):

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

    # scale
    test_features = test_features.astype('float32')/255
    test_features = datagen.standardize(test_features)

    # predict
    result = model.predict(test_features)

    # result
    predicted_class = np.argmax(result, axis=1).astype(int)
    num_correct = np.sum(predicted_class == test_sample_labels)
    accuracy = float(num_correct)/result.shape[0]

    # save
    filename = 'results'+extra_name+'.csv'
    np.savetxt(filename, predicted_class, fmt='%d')

    print 'Accuracy (compare with sample output)', accuracy * 100

def main():

    # load json and create model
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    # load test data
    test_features, test_sample_labels = load_cifar3_test()

    # predict
    save_predict(test_features, test_sample_labels, model, '')

if __name__ == "__main__":
    main()
