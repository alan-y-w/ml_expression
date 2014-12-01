import scipy.io as sio
from defs import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv
from scipy import ndimage
import preprocessing as prep



"""

"""
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
    valid_targets, valid_ids, valid_data, valid_keys = create_sub_set(persons, valid_count)
#     valid_data = edge_detect(valid_data)



    # remove the persons used for validation set
    for key in valid_keys:
        del persons[key]
#     print valid_targets.shape, valid_ids.shape, valid_data.shape

    # get training set
    train_targets, train_ids, train_data, train_keys = create_sub_set(persons, train_count)
#     train_data = edge_detect(train_data)
    return valid_targets, valid_ids, valid_data.T, \
           train_targets.T.reshape(train_targets.shape[1]), train_ids, train_data.T

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
                    data = preprocess_image(data)
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
                else:
                    while (i > 0):
                        targets = np.delete(targets, -1)
                        ids = np.delete(ids,-1)
                        data = np.delete(data, -1, axis=0)
                        i -= 1
#                     keys.append(key)

                    data = preprocess_image(data)
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
        keys.append(key)
    # in this case all persons finished
    # in case count >> the number of samples, remove zeros in the return arrays
    while(counter < count - 1):
        targets = np.delete(targets, -1)
        ids = np.delete(ids, -1)
        data = np.delete(data, -1, axis=0)
        counter += 1

    data = preprocess_image(data)
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

                    data = preprocess_image(data)
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
                else:
                    while (i > 0):
                        targets = np.delete(targets, -1)
                        ids = np.delete(ids,-1)
                        data = np.delete(data, -1, axis=0)
                        i -= 1
#                     keys.append(key)

                    data = preprocess_image(data)
                    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys
        keys.append(key)
    # in this case all persons finished
    # in case count >> the number of samples, remove zeros in the return arrays
    while(counter < count - 1):
        targets = np.delete(targets, -1)
        ids = np.delete(ids, -1)
        data = np.delete(data, -1, axis=0)
        counter += 1
    data = preprocess_image(data)
    return targets.reshape(1, targets.shape[0]), ids.reshape(1, ids.shape[0]), data, keys


"""
Calculate the difference between the two model errors.(Common error rate)
input:
    model1: predictions from prediction method 1, 1 x n array
    model2: predictions from prediction method 2, 1 x n array
"""
def compare_classification_err(model1, model2, predictions):
    err_1 = model1 - predictions
    err_2 = model2 - predictions
    total_err = 0.
    common_err = 0.
    tot_err_1 = 0.

    for i in xrange(predictions.shape[1]):
        if err_1[0][i] != 0 or err_2[0][i] != 0:
            total_err += 1
            if err_1[0][i] != 0:
                tot_err_1 += 1
            if err_1[0][i] != 0 and err_2[0][i] != 0:
                common_err +=1
    print "Total error count of both model is %.f \n" \
          "Common error rate: %.5f\n" \
          "Percentage of model 1 error: %.2f\n"%(total_err, common_err/total_err, tot_err_1/(common_err + total_err))

"""
input:
    predictions: (#samples, ) array of predictions
    targets: (1, #samples) array of targets
output:
    percentage error
"""
def percent_error(predictions, targets):
    error_count = np.count_nonzero(predictions - targets)
    return float(error_count)/predictions.shape[0]

"""
Save the test prediction result into csv
"""
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

"""
resize and show images from vector.
input:
        images: n x (h * w)
                h * w = 1024
        n_image: number of faces you want to show
        k: the number of the figure.
"""
def show_image(images, n_image, k=1):

    print images.shape
    plt.figure(k)
    for i in range(n_image):
        plt.subplot(1, n_image, i+1)
        plt.imshow(images[i,:].reshape([32, 32]).T, cmap='gray', interpolation='nearest')
    plt.draw()
    plt.show()

"""
Reshape and Normalize the image vector so that it is easier for computation.

Input:
    image_data: M, h, w

Output:
    image_data: h*w, M
"""
def preprocess_image(image_data):

    # Reshape data vector to M x (hxw).
    M, h, w  = image_data.shape
    image_data = image_data.reshape(M, h*w).T
    #
    # Preprocess the image
    image_data = preprocessing.scale(np.float32(image_data))
    # return prep.high_pass(image_data, M)
    return image_data

def localize_image(data, scale):
    pass

if __name__ == '__main__':
    n_image = 5
    # original faces.
    images = load_data_test("public_test_images").T
    print images.shape
    show_image(images, n_image, 2)
    
    # ZCA image processing.
    # images = ZCA(images)
    # show_image(images, n_image, 3)
    
    # high pass processing.
    prep.high_pass(images.T, n_image)
    show_image(images, n_image, 3)
