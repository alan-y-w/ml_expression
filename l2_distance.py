import numpy as np
from l2_distance import l2_distance

def run_knn(k, train_data, train_labels, test_data):
  """Uses the supplied training inputs and labels to make
  predictions for test data using the K-nearest neighbours
  algorithm.

  Note: M is the number of training examples,
        N is the number of test examples, 
        and F is the number of features per example.

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
  M = train_data.shape()[1]
  N = test_data.shape()[1]
  prediction = np.zeros(N, dtype=numpy.int)

  dist = l2_distance(valid_data, train_data)
  nearest = np.argsort(dist, axis=1)[:,:k]

  train_labels = train_labels.reshape(-1)
  test_labels = train_labels[nearest]

  for i in xrange(ntest):
    K = k
    while True:
      hist, edge = np.histogram(test_labels[i,:])
      best = hist.index(hist.max())
      np.sort(hist)
      if (hist[-1] > hist[-2] or k == M):
        break
      else:
        k += 1
    prediction[i] = best
  return prediction

def shape_image(tr_image, te_image):
  h, w, ntr = tr_image.shape()
  ntest = te_image.shape()[2]
  tr_image = tr_image.reshape(h*w, ntr)
  te_image = tr_image.reshape(h*w, ntest)


def normalize_image(tr_image, te_image):
  tr_mu = np.mean(tr_image)
  te_mu = np.mean(te_image)
  tr_image = tr_image - tr_mu
  te_image = te_image - te_mu

  tr_std = np.std(tr_image)
  te_std = np.std(te_image)
  # if std != 0
  tr_image = tr_image / tr_std
  te_image = te_image / te_std

def main():
  K = 5
  # Change this line to load numbers
  tr_image, va_image, te_image, tr_labels, va_labels, te_labels = LoadData('digits.npz')
  tr_image, te_image = shape_image(tr_image, te_image)
  tr_image, te_image = normalize_image(tr_image, te_image)
  run_knn(K, tr_image, tr_labels, te_image)
