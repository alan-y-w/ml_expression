import numpy as np
from util import *
from pcaimg import *

def main():
    # make sure valid_count + train_count
    valid_size = 96
    train_size = 2830
    valid_targets, valid_ids, valid_data, \
    train_targets, train_ids, train_data = load_data(valid_size, train_size);
    print valid_targets.shape, valid_ids.shape, valid_data.shape
    print train_targets.shape, train_ids.shape, train_data.shape

    # (pixels as array, #num of samples)
    # train target: (num_sampels, height, width)
    # must reshape arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2]).T

    K = 256
    print train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).shape
    print train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).T.shape
    raw_input("Meh")
    v, mean, projX, w = pcaimg(train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).T, K)

    print train_data[0]
    ShowEigenVectors(v.T[0:3].T)
    ShowMeans(mean)

if __name__ == '__main__':
  main()
