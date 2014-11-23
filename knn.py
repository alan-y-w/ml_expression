import numpy as np
from util import *

def run_knn(K, train_data, train_labels, test_data):
  """Uses the supplied training inputs and labels to make
  predictions for test data using the K-nearest neighbours
  algorithm.

  Note: M is the number of training examples,
        N is the number of test examples,
        and F is the number of pixels per image.

  Inputs:
      k:            The number of neighbours to use for classification 
                    of a test example.
      train_data:   The F x M array of training
                    data.
      train_labels: The M x 1 vector of training labels
                    corresponding to the examples in train_data 
                    (must be binary).
      test_data:   The F x N array of data to
                    predict classes for.

  Outputs:
      test_labels: The N x 1 vector of predicted labels 
                    for the test data.
  """

  M = train_data.shape[1]
  N = test_data.shape[1]
  prediction = np.zeros(N)

  dist = l2_distance(test_data, train_data)
  nearest = np.argsort(dist, axis=1)[:,:]

  train_labels = train_labels.reshape(-1)
  test_labels = train_labels[nearest]

  for i in xrange(N):
    k = K
    while True:
      hist, edge = np.histogram(test_labels[i,:k], bins=[1,2,3,4,5,6,7,8])
      best = np.where(hist == hist.max())[0]

      if (best.size == 1 or k == M):
        break
      else:
        k += 1
    prediction[i] = best[0] + 1
  return prediction

def shape_image(tr_image, te_image):
  """
  Shape the images samples to a two dimensional vector.

  Input:
        tr_image: M x h x w
        te_image: N x h x w 

  Output:
        tr_image: (h x w) x M
        te_image: (h x w) x N
  """
  M, h, w = tr_image.shape
  N = te_image.shape[0]
  tr_image = tr_image.reshape(M, h*w).T
  te_image = te_image.reshape(N, h*w).T
  return tr_image, te_image 


def normalize_image(tr_image, te_image):
  """
  Normalize the image vector so that it is easier for computation.
  """
  tr_image = tr_image - np.mean(tr_image)
  te_image = te_image - np.mean(te_image)

  tr_std = np.std(tr_image)
  te_std = np.std(te_image)
  if tr_std != 0 and te_std != 0:
    tr_image = tr_image / tr_std
    te_image = te_image / te_std
    return tr_image, te_image
  else:
    print "Warning: There are no differences among input images."

def l2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.

    Inputs:
        A: D x M array.
        B: D x N array.

    Returns:
        E: M x N Euclidean distances between vectors in A and B.
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)

def validate_image(validate_lebals, predictions):
  if validate_lebals.shape == predictions.shape:
    correct = 0.0
    N = validate_lebals.shape[0]
    for i in xrange(N):
      if validate_lebals[i] == predictions[i]:
        correct += 1
    return correct / N

  else:
    return "Warning: Two sets are not in the same size"

def main():

  # Change this line to load numbers
  valid_size = 96
  train_size = 2830
  te_labels, valid_ids, te_image, \
  tr_labels, train_ids, tr_image = load_data(valid_size, train_size)
  print te_labels.shape, valid_ids.shape, te_image.shape
  print tr_labels.shape, train_ids.shape, tr_image.shape
  
  # Change this line to cange KNN.
  K = 5
  tr_image, te_image = shape_image(tr_image, te_image)
  tr_image, te_image = normalize_image(tr_image, te_image)
  predictions = run_knn(K, tr_image, tr_labels, te_image)
  print validate_image(te_labels, predictions)

if __name__ == '__main__':
  main()
