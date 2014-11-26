import scipy.io as sio
from defs import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv
from scipy import ndimage
from loaddata import *

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
#         plt.imshow(image.T, cmap="gray")
#         plt.show()

        lowpass = ndimage.gaussian_filter(image, 3)
        image = image - lowpass


        data[i] = image.reshape(data[i].shape)
#         plt.imshow(data[i].reshape(32,32).T, cmap="gray")
#         plt.show()

#         sx = ndimage.sobel(image, axis=0, mode='constant')
#         sy = ndimage.sobel(image, axis=1, mode='constant')
#         sob = np.hypot(sx, sy)
#         ShowMeans(sob.reshape(1024,1))
#         data[i] = sob.reshape(data[i].shape)

    data = preprocessing.scale(data)
    return data.T

def preprocess_image(image_data):
    """
    Reshape and Normalize the image vector so that it is easier for computation.

    Input:
          image_data: M, h, w

    Output:
          image_data: h*w, M
    """
    # Reshape data vector to M x (hxw).
    h, w, M = image_data.T.shape
    image_data = image_data.T.reshape(h*w, M)
    return preprocessing.scale(np.float32(image_data))

def ShowMeans(means):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in xrange(means.shape[1]):
        plt.subplot(1, means.shape[1], i+1)
        plt.imshow(means[:, i].reshape(32, 32).T, cmap="gray")
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

    lowpass = ndimage.gaussian_filter(images[1], 2)
    gauss_highpass = images[1] - lowpass
    plt.imshow(preprocessing.scale(np.float32(gauss_highpass.T)), cmap='gray')
    plt.show()

    image = sob
    image = preprocessing.scale(np.float32(image))
#     ShowMeans(gauss_highpass.reshape(1024,1))

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
