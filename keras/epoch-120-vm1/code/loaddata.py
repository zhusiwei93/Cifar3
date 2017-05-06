import numpy as np

def show_image(image):

    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def load_cifar3_train():

    print('Loading data from disk')
    raw = np.fromfile('../data_bin/data_batch.bin',dtype=np.dtype('uint8'))

    mat = raw.reshape(12000,3073)
    labels = mat[:,0].reshape(12000,1)
    features = mat[:,1:].reshape(12000,3,32,32)

    return features, labels

def load_cifar3():

    print('Loading data from disk')
    raw = np.fromfile('../data_bin/data_batch.bin',dtype=np.dtype('uint8'))

    mat = raw.reshape(12000,3073)

    # shuffle
    np.random.shuffle(mat)

    labels = mat[:,0]
    features = mat[:,1:].reshape(12000,3,32,32)
    pivot = 10000

    train_features = features[:pivot]
    train_labels = labels[:pivot].reshape(pivot,1)

    test_features = np.array(features)[pivot:]
    test_labels = np.array(labels)[pivot:].reshape(12000-pivot,1)

    # show_image(train_features[203][2])
    return (train_features, train_labels), (test_features, test_labels)

def load_cifar3_test():

    print('Loading data from disk')
    raw = np.fromfile('../data_bin/test_data.bin',dtype=np.dtype('uint8'))

    features = raw.reshape(3000,3,32,32)
    labels = np.loadtxt('../results.csv')

    return features, labels
