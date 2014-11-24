import numpy as np
from util import *
from sklearn import svm
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

""" WRAPPER FUNCTIONS """

"""
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_pca_svm(train_data, train_targets, test_data, test_targets, pca_count=80):
    ###############################################################################
    # PCA
    n_components = pca_count
#     n_components = 5
    #pca.fit = shape (n_samples, n_features)
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(train_data)

    ###############################################################################
    # SVM
    # enabel probability estimate to use with adaboost
    svm_clf = svm.SVC(probability=False)
    # train_data.shape = (pixels, #samples)
    # train_targets.shape = (1, #samples)

    # transform data
    train_data_pca = pca.transform(train_data)
    valid_data_pca = pca.transform(test_data)
    # reshape targets for fitting


    # can also get probability score
    svm_clf.fit(train_data_pca, train_targets)
    valid_predictions = svm_clf.predict(valid_data_pca)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    return svm_clf, (1 - percent_error(valid_predictions, test_targets))

"""
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_random_forest(train_data, train_targets, test_data, test_targets, num_tree=200):
    # random forest (works better without pca)
    trees_clf = RandomForestClassifier(n_estimators=num_tree)
    trees_clf = trees_clf.fit(train_data, train_targets)
#     clf_trees = clf_trees.fit(train_data_pca, train_targets_reshaped)

    # can also get probability score
    valid_predictions = trees_clf.predict(test_data)
#     valid_predictions = clf_trees.predict(valid_data_pca)

    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    return trees_clf, (1 - percent_error(valid_predictions, test_targets))

"""
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_adaboost(train_data, train_targets, test_data, test_targets, num_learner=50):
    ada_clf = AdaBoostClassifier(n_estimators=num_learner)
    ada_clf = ada_clf.fit(train_data, train_targets)
#     scores = cross_val_score(ada_clf, valid_data_pca, valid_targets)
#     scores.mean()

    valid_predictions = ada_clf.predict(test_data)
#     print valid_predictions.shape, valid_targets.shape
#     valid_predictions.shape = (num_samples,)
    return ada_clf, 1 - percent_error(valid_predictions.reshape(1, valid_predictions.shape[0]), test_targets)


def main():
    # make sure valid_count + train_count
    n = 2
    valid_size = 96  + 96 * n
    train_size= 2830 - 96 * n
    filename = 'labeled_images.mat'
    valid_targets, valid_ids, valid_data, \
    train_targets, train_ids, train_data = load_data(valid_size, train_size, filename);
#     print valid_targets.shape, valid_ids.shape, valid_data.shape
#     print train_targets.shape, train_ids.shape, train_data.shape

    #reshape data and labels
    train_data = train_data.T
    valid_data = valid_data.T
    train_targets = train_targets.T.reshape(train_targets.shape[1])
    valid_targets = valid_targets.T.reshape(valid_targets.shape[1])

    ###############################################################################
    # PCA-SVM

    # 80 components gives the best result with SVM
    pca_count = 80
    _, hit_rate = wrapper_pca_svm(train_data, train_targets, valid_data, valid_targets, pca_count)
    print ("PCA-SVM hit rate: %.5f" % hit_rate)

    ###############################################################################
    # Random forest
    # 200 trees give the best result
#     num_tree = 200
#     _, hit_rate = wrapper_random_forest(train_data, train_targets, valid_data, valid_targets, num_tree)
#     print ("random forest hit rate: %.5f" % hit_rate)

    ###############################################################################
    # adaboost
    num_learner = 20
    _, hit_rate = wrapper_random_forest(train_data, train_targets, valid_data, valid_targets, num_learner)
    print ("adaboosted pca-svm hit rate: %.5f" % hit_rate)

if __name__ == '__main__':
    main()

