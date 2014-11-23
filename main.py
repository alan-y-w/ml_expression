import numpy as np
from util import *
from sklearn import svm
from sklearn.decomposition import RandomizedPCA

def main():
    # make sure valid_count + train_count
    n = 2
    valid_size = 96  + 96 * n
    train_size= 2830 - 96 * n
    filename = 'labeled_images.mat'
    valid_targets, valid_ids, valid_data, \
    train_targets, train_ids, train_data = load_data(valid_size, train_size, filename);
    print valid_targets.shape, valid_ids.shape, valid_data.shape
    print train_targets.shape, train_ids.shape, train_data.shape

    ###############################################################################
    # PCA
    n_components = 80
    #pca.fit = shape (n_samples, n_features)
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(train_data.T)

#     # transform data
#     X_train_pca = pca.transform(X_train)
#     valid_data_pca = pca.transform(X_test)


    ###############################################################################
    # SVM
    clf = svm.SVC()
    # train_data.shape = (pixels, #samples)
    # train_targets.shape = (1, #samples)

    # transform data
    train_data_pca = pca.transform(train_data.T)
    valid_data_pca = pca.transform(valid_data.T)
    # reshape targets for fitting
    train_targets_reshaped = train_targets.T.reshape(train_targets.shape[1])

    clf.fit(train_data_pca, train_targets_reshaped)
    valid_predictions = clf.predict(valid_data_pca)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])

    print (1 - percent_error(valid_predictions, valid_targets))


    # (pixels as array, #num of samples)
    # train target: (num_sampels, height, width)
    # must reshape arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2]).T

#     K = 256
#     print train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).shape
#     print train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).T.shape
#     raw_input("Meh")
#     v, mean, projX, w = pcaimg(\
#                     train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2]).T, K)

#     ShowEigenVectors(v.T[0:3].T)
#     ShowMeans(mean)




if __name__ == '__main__':
  main()
