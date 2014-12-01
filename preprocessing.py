from util import *
from sklearn import preprocessing


"""
process image so that high contrast gets passed.
input:
        images: (h * w) x n
                h * w = 1024
        n_image: number of faces you want to process.
output:
        images: (h * w) x n
"""
def high_pass(images, n_image):

    for i in range(n_image):
        lowpass = ndimage.gaussian_filter(images[:,i].reshape([32, 32]), 3)
        gauss_highpass = images[:,i] - lowpass.reshape([1024,])
        images[:,i] = gauss_highpass
    return images

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
#     show_means(preprocessing.scale(np.float32(image)))

#     print images[1].shape
    sx = ndimage.sobel(images[1], axis=0, mode='constant')
    sy = ndimage.sobel(images[1], axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    image = sob
    image = preprocessing.scale(np.float32(image))
    # show_means(image.reshape(1024,1))
