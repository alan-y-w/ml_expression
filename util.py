import scipy.io as sio
from defs import *
import matplotlib.pyplot as plt
from sklearn import preprocessing

def load_data(valid_count, train_count, filename):
    persons = {};

    mat = sio.loadmat(filename)
    labels = mat['tr_labels'] #(2925, 1) examples
    #print labels
    images = mat['tr_images'].T #(32, 32, 2925)
    ids = mat['tr_identity'] #(2925, 1)

#     print images[0]
#     ShowMeans(images[0].reshape(1, 1024).T)

#     ShowMeans(images[0].reshape(1, 1024).T)
#     ShowMeans(preprocessing.scale(np.float32(images[0].reshape(1, 1024).T)))


    # collect data for each individual
    for i in xrange(labels.shape[0]):
        p_id = ids[i][0]
        if (not persons.has_key(p_id)):
            persons[p_id] = Person(p_id)
        persons[p_id].data.append(images[i])
        persons[p_id].expressions.append(labels[i][0])

    # split into training set and validation set
    # 743 persons
    valid_targets, valid_ids, valid_data, valid_keys = create_sub_set(persons, valid_count)

    # remove the persons used for validation set
    for key in valid_keys:
        del persons[key]
#     print valid_targets.shape, valid_ids.shape, valid_data.shape

    # get training set
    train_targets, train_ids, train_data, train_keys = create_sub_set(persons, train_count)

    return valid_targets, valid_ids, valid_data, train_targets, train_ids, train_data
#     print train_targets.shape, train_ids.shape, train_data.shape

"""
Take the persons dictionary and takes out samples for the set
Also return a list of keys of the persons used for set
"""
def create_sub_set(persons, count):

    targets = np.zeros((count))
    ids = np.zeros((count))
    data = np.zeros((count, 32, 32))
    keys = []
    counter = 0

    for key in persons:
        for i in range(len(persons[key].expressions)):
            if (counter < count):
                targets[counter] = persons[key].expressions[i]
                ids[counter] = persons[key].id

                data[counter] = persons[key].data[i]

                counter += 1
            else:
                if (i == 0):
                    keys.append(key)

                    n, h, w = data.shape
                    data = data.reshape(n, (h*w)).T
                    data = preprocessing.scale(np.float32(data))
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
                else:
                    while (i > 0):
                        targets = np.delete(targets, -1)
                        ids = np.delete(ids,-1)
                        data = np.delete(data, -1, axis=0)
                        i -= 1
#                     keys.append(key)

                    n, h, w = data.shape
                    data = data.reshape(n, (h*w)).T
                    data = preprocessing.scale(np.float32(data))
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
        keys.append(key)
    # in this case all persons finished
    # in case count >> the number of samples, remove zeros in the return arrays
    while(counter < count - 1):
        targets = np.delete(targets, -1)
        ids = np.delete(ids, -1)
        data = np.delete(data, -1, axis=0)
        counter += 1

    n, h, w = data.shape
    data = data.reshape(n, (h*w)).T
    data = preprocessing.scale(np.float32(data))
    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys

def ShowMeans(means):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in xrange(means.shape[1]):
        plt.subplot(1, means.shape[1], i+1)
        plt.imshow(means[:, i].reshape(32, 32).T, cmap=plt.cm.gray)
    plt.draw()
    raw_input('Press Enter.')

"""
image_array.shape = (number of images, 32, 32)
return.shape = (32x32, number of images)
"""
def image_reshape(image_array):
    return image_array.reshape(image_array.shape[0], image_array.shape[1] * image_array.shape[2]).T

"""
input:
    predictions: (1, #samples) array of predictions
    targets: (1, #samples) array of targets
"""
def percent_error(predictions, targets):
    error_count = np.count_nonzero(predictions - targets)
    return float(error_count)/predictions.shape[1]

#code to visualize the image
# def ShowImage():

#     print images[0].shape
#     imgplot=plt.imshow(images[0].T)
#     plt.show()
#     raw_input("Enter")