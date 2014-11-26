import scipy.io as sio
from defs import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv
from scipy import ndimage

"""
input:
    data: (h * w, #samples)
output:
    data: (h * w, #samples)
"""
def edge_detect(data):
    # pre-process valid_data
    data = data.T

    for i in xrange(data.shape[0]):
        image = data[i]
        image = image.reshape(32,32)
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
#         ShowMeans(sob.reshape(1024,1))
        data[i] = sob.reshape(data[i].shape)

    data = preprocessing.scale(data)
    return data.T

def load_data(valid_count, train_count, filename):


    mat = sio.loadmat(filename)
    labels = mat['tr_labels'] #(2925, 1) examples
    #print labels
    images = mat['tr_images'].T #(32, 32, 2925)
    ids = mat['tr_identity'] #(2925, 1)

#     print images[0]
#     ShowMeans(images[0].reshape(1, 1024).T)

#     ShowMeans(images[0].reshape(1, 1024).T)
#     ShowMeans(preprocessing.scale(np.float32(images[0].reshape(1, 1024).T)))

    persons = create_persons(labels, images, ids)


    # split into training set and validation set
    # 743 persons
    valid_targets, valid_ids, valid_data, valid_keys = create_sub_set_rand(persons, valid_count)
#     valid_data = edge_detect(valid_data)



    # remove the persons used for validation set
    for key in valid_keys:
        del persons[key]
#     print valid_targets.shape, valid_ids.shape, valid_data.shape

    # get training set
    train_targets, train_ids, train_data, train_keys = create_sub_set_rand(persons, train_count)
#     train_data = edge_detect(train_data)
    return valid_targets, valid_ids, valid_data, train_targets, train_ids, train_data
"""
input:
    file name to load the test image
output:
    array of test data (h*w, #samples)
"""
def load_data_test(filename):
#     #['__version__', 'public_test_images', '__header__', '__globals__']
#     #  test_data.shape = (32, 32, 418)
    mat = sio.loadmat(filename)
    #print labels
    images = mat['public_test_images'].T #(32, 32, 2925)
    ids =  np.arange(images.shape[0])
    ids = ids.reshape((ids.shape[0], 1))
    labels = np.zeros(ids.shape)#(2925, 1) examples

#     print images[0]
#     ShowMeans(images[0].reshape(1, 1024).T)

#     ShowMeans(images[0].reshape(1, 1024).T)
#     ShowMeans(preprocessing.scale(np.float32(images[0].reshape(1, 1024).T)))

    persons = create_persons(labels, images, ids)


    # split into training set and validation set
    # 743 persons
    _, _, test_data, _ = create_sub_set(persons, labels.shape[0])
#     test_data = edge_detect(test_data)
    return test_data


def load_test(filename):
    mat = sio.loadmat(filename)
    test_image = mat['public_test_images']
    return preprocess_image(test_image)


def preprocess_image(image_data):
  """
  Reshape and Normalize the image vector so that it is easier for computation.

  Input:
        image_data: h, w, M

  Output:
        image_data: h*w, M
  """
  # Reshape data vector to M x (hxw).
  h, w, M = image_data.shape
  image_data = image_data.reshape(h*w, M)
  return preprocessing.scale(np.float32(image_data))

#   # Start to normalize_image
#   image_data = image_data - np.mean(image_data)
#
#   tr_std = np.std(image_data)
#   if tr_std != 0:
#     image_data = image_data / tr_std
#     return image_data
#   else:
#     print "Warning: There are no differences among input images."

"""
input:
    labels, ids: (#samples, 1) array of labeles and ids
    data: (h, w, samples) array of data
output:
    percentage error
"""
def create_persons(labels, images, ids):
    persons = {};
    # collect data for each individual
    for i in xrange(labels.shape[0]):
        p_id = ids[i][0]
        if (not persons.has_key(p_id)):
            persons[p_id] = Person(p_id)
        persons[p_id].data.append(images[i])
        persons[p_id].expressions.append(labels[i][0])
    return persons

"""
Take the persons dictionary and takes out first "N" samples for the set
Also return a list of keys of the persons used for set

Note that the number of samples returned may be less than count, as the data about
the same person must stay in the same set. There is not splitting

input:
    persons: dictionary of persons, by id
    count: MAXIMUM number of samples in the returned set

output:
    subset of data and keys used to extract the data
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

"""
Take the persons dictionary and takes out random "N" samples for the set
Also return a list of keys of the persons used for set

Note that the number of samples returned may be less than count, as the data about
the same person must stay in the same set. There is not splitting

input:
    persons: dictionary of persons, by id
    count: MAXIMUM number of samples in the returned set

output:
    subset of data and keys used to extract the data
"""
def create_sub_set_rand(persons, count):
    targets = np.zeros((count))
    ids = np.zeros((count))
    data = np.zeros((count, 32, 32))
    keys = []
    counter = 0

    # construct keys
    shuffle_keys = persons.keys()
    np.random.shuffle(shuffle_keys)

    for key in shuffle_keys:
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
    
def preprocess_image(data):
    """
    A function preprocess image with repect to different algorithms.
    input:
            data: n x h x w
    output: 
            data: (h * w) x n
    """
    n, h, w = data.shape
    data = data.reshape(n, (h*w)).T
    data = preprocessing.scale(np.float32(data))
    data = high_pass(data, n)
    return data

def ShowMeans(means):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in xrange(means.shape[1]):
        plt.subplot(1, means.shape[1], i+1)
        plt.imshow(means[:, i].reshape(32, 32).T, cmap=plt.cm.gray)
    plt.draw()
    plt.show()

"""
Reshape the image as array
* image_array.shape = (number of images, h, w)
* return.shape = (hxw, number of images)
"""
def image_reshape(image_array):
    return image_array.reshape(image_array.shape[0], image_array.shape[1] * image_array.shape[2]).T

"""
input:
    predictions: (1, #samples) array of predictions
    targets: (1, #samples) array of targets
output:
    percentage error
"""
def percent_error(predictions, targets):
    error_count = np.count_nonzero(predictions - targets)
    return float(error_count)/predictions.shape[1]


def save_csv(result, filename="submission.csv"):
    """save all the results into a csv file."""
    with open(filename, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Id", "Prediction"])
        writer.writeheader()
        for i in xrange(result.shape[1]):
            writer.writerow({"Id":i+1, "Prediction":int(result[0][i])})

        i+=1
        while (i < 1253):
            i+=1
            writer.writerow({"Id":i, "Prediction":0})
# """
# input:
#     predictions: (1, #samples) array of predictions
#     targets: (1, #samples) array of targets
# output:
#     percentage error
# """
# def split_set_in_half(data, targets, ids):


#code to visualize the image
# def ShowImage():

#     print images[0].shape
#     imgplot=plt.imshow(images[0].T)
#     plt.show()
#     raw_input("Enter")

def feature_extraction():
    filename = "public_test_images"
    mat = sio.loadmat(filename)
    #print labels
    images = mat['public_test_images'].T #(32, 32, 2925)
    ids =  np.arange(images.shape[0])
    ids = ids.reshape((ids.shape[0], 1))
    labels = np.zeros(ids.shape)#(2925, 1) examples

    print "images shape: %s" % str(images.shape)
    image = images[1].reshape(1024, 1)

#     ShowMeans(image)
    ShowMeans(preprocessing.scale(np.float32(image)))

#     print images[1].shape
    sx = ndimage.sobel(images[1], axis=0, mode='constant')
    sy = ndimage.sobel(images[1], axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    image = sob
    image = preprocessing.scale(np.float32(image))
    ShowMeans(image.reshape(1024,1))

def show_image(images, n_image, k):
    """
    resize and show images from vector.
    input: 
            images: n x (h * w)
                    h * w = 1024
            n_image: number of faces you want to show
            k: the number of the figure.
    """
    plt.figure(k)
    for i in range(n_image):
        plt.subplot(1, n_image, i+1)
        plt.imshow(images[i,:].reshape([32, 32]).T, cmap='gray', interpolation='nearest')
    plt.draw()

def high_pass(images, n_image):
    """
    process image so that high contrast gets passed.
    input: 
            images: (h * w) x n
                    h * w = 1024
            n_image: number of faces you want to process.
    """
    for i in range(n_image):
        lowpass = ndimage.gaussian_filter(images[:,i].reshape([32, 32]), 2)
        gauss_highpass = images[:,i] - lowpass.reshape([1024,])
        images[:,i] = gauss_highpass
    return images

if __name__ == '__main__':
    n_image = 5
    # original faces.
    images = load_data_test("public_test_images").T
    show_image(images, n_image, 2)
    
    # ZCA image processing.
    # images = ZCA(images)
    # show_image(images, n_image, 3)
    
    # high pass processing.
    high_pass(images.T, n_image)
    show_image(images, n_image, 2)
    
    plt.show()

    feature_extraction()
