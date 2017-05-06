import numpy as np
from loaddata import load_cifar3_test
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def accuracy(test_features, test_labels, model):

    result = model.predict(test_features)
    predicted_class = np.argmax(result, axis=1).astype(int)
    true_class = np.argmax(test_labels)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]

    np.savetxt('predict.csv', predicted_class, fmt='%d')
    return (accuracy * 100)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_features, test_labels = load_cifar3_test()
print 'Accuracy', accuracy(test_features, test_labels, loaded_model)
