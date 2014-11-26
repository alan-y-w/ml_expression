import numpy as np
from util import *
from loaddata import *
from sklearn import svm
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier

""" ########### WRAPPER FUNCTIONS ########### """

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
#     print test_data.shape
    # transform data
    train_data_pca = pca.transform(train_data)
    valid_data_pca = pca.transform(test_data)
    # reshape targets for fitting


#     # can also get probability score
    svm_clf.fit(train_data_pca, train_targets)
    valid_predictions = svm_clf.predict(valid_data_pca)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    return svm_clf, pca, (1 - percent_error(valid_predictions, test_targets))


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

    ada_clf = AdaBoostClassifier(n_estimators=num_learner,learning_rate=0.2)
    ada_clf = ada_clf.fit(train_data, train_targets)
#     scores = cross_val_score(ada_clf, valid_data_pca, valid_targets)
#     scores.mean()

    valid_predictions = ada_clf.predict(test_data)
#     print valid_predictions.shape, valid_targets.shape
#     valid_predictions.shape = (num_samples,)
    return ada_clf, 1 - percent_error(valid_predictions.reshape(1, valid_predictions.shape[0]), test_targets)

"""
input:
    arrary_of_predictions: array of (#samples, class), each item is a probability

output:
    prediction: (#samples, )
"""
def max_proba_predictions(arrary_of_predictions):
    proba_product = np.ones(arrary_of_predictions[0].shape)
    ret_predictions = np.zeros(arrary_of_predictions[0].shape[0])

    # take product of all the class probabilities
    for prediction in arrary_of_predictions:
        # normailize the probabilities to the same confidence level
#         for i in xrange(prediction.shape[0]):
#             max = np.amax(prediction[i])
#             prediction[i] = prediction[i] / max
        proba_product = proba_product * prediction

    # select the maximum probability as class prodiction
    for i in xrange(proba_product.shape[0]):
        ret_predictions[i] = np.argmax(proba_product[i])
    return ret_predictions

""" ########### TOP FUNCTIONS ########### """
"""
    Code to experiment different classifier
"""
def main_single_classifier():
    # make sure valid_count + train_count
    n = 3
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
    svm, pca, hit_rate = wrapper_pca_svm(train_data, train_targets, valid_data, valid_targets, pca_count)
    print ("PCA-SVM hit rate: %.5f" % hit_rate)

    ###############################################################################
    # Random forest

    # 200 trees give the best result
#     num_tree = 200
#     rand_tree, hit_rate = wrapper_random_forest(train_data, train_targets, valid_data, valid_targets, num_tree)
#     print ("random forest hit rate: %.5f" % hit_rate)

    ###############################################################################
    # Adaboost

#     num_learner = 600
#     _, hit_rate = wrapper_adaboost(pca.transform(train_data), train_targets,\
#                                     pca.transform(valid_data), valid_targets, num_learner)
#     print ("adaboosted pca-svm hit rate: %.5f" % hit_rate)

    ###############################################################################
    # Extra trees
#     n = [40, 60, 80, 100, 120, 140, 180, 200]
#     for num in n:
#         print num
#         extraTrees = ExtraTreesClassifier(n_estimators=num)
#         extraTrees.fit((train_data), train_targets)
#         valid_predictions = extraTrees.predict((valid_data))
#     #     print valid_predictions.shape, valid_targets.shape
#     #     valid_predictions.shape = (num_samples,)
#         print "extraTrees hit rate: %.5f" % \
#         (1 - percent_error(valid_predictions.reshape(1, valid_predictions.shape[0]), valid_targets))
    ###############################################################################
    #
#     n = [40, 60, 80, 100, 120, 140, 180, 200]
#     for num_tree in n:
#         print num_tree
#         trees_clf = RandomForestClassifier(n_estimators=num_tree)
#         trees_clf = trees_clf.fit(pca.transform(train_data), train_targets)
#     #     clf_trees = clf_trees.fit(train_data_pca, train_targets_reshaped)
#         # can also get probability score
#         valid_predictions = trees_clf.predict(pca.transform((valid_data)))
#     #     valid_predictions = clf_trees.predict(valid_data_pca)
#
#         valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
#         print (1 - percent_error(valid_predictions, valid_targets))

    ###############################################################################
    # Generate test result
    # alan's code
#     test_images = load_data_test("public_test_images").T
#     print test_images.shape
#     test_data_pca = pca.transform(test_images)
#     test_predictions = svm.predict(test_data_pca)
#     test_predictions = test_predictions.reshape(1, test_predictions.shape[0])
#     save_csv(test_predictions)
#     print "Done writing CSV file!"

"""
    Use log probabilities from different classifier to choose output class
"""
def main_max_logprob():
    # make sure valid_count + train_count
    n = 3
    valid_size = 50
    train_size= 3000
    filename = 'labeled_images.mat'
    valid_targets, valid_ids, valid_data, \
    train_targets, train_ids, train_data = load_data(valid_size, train_size, filename);

    #reshape data and labels
    train_data = train_data.T
    valid_data = valid_data.T
    train_targets = train_targets.T.reshape(train_targets.shape[1])
    valid_targets = valid_targets.T.reshape(valid_targets.shape[1])

    # train PCA SVM
    # PCA
    n_components = 80
#     n_components = 5
    #pca.fit = shape (n_samples, n_features)
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(train_data)

    ###############################################################################
    # SVM
    # enable probability estimate to use with adaboost
    svm_clf = svm.SVC(probability=True)

    # transform data
    train_data_pca = pca.transform(train_data)
    valid_data_pca = pca.transform(valid_data)
    # reshape targets for fitting


    # can also get probability score
    svm_clf.fit(train_data_pca, train_targets)
    print "Num Train Samples: %d" % train_targets.shape
    valid_prob_svm = svm_clf.predict_proba(valid_data_pca)

    valid_predictions = svm_clf.predict(valid_data_pca)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    hit_rate_svm = 1 - percent_error(valid_predictions, valid_targets)
    print("SVM: %.5f" % (hit_rate_svm))

    print valid_prob_svm.shape

    ###############################################################################
    # random forest (works better without pca)
    trees_clf = RandomForestClassifier(n_estimators=200)
    trees_clf = trees_clf.fit(train_data, train_targets)
#     clf_trees = clf_trees.fit(train_data_pca, train_targets_reshaped)
    # can also get probability score
    valid_prob_forests = trees_clf.predict_proba(valid_data)

    valid_predictions = trees_clf.predict(valid_data)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    hit_rate_forests = 1 - percent_error(valid_predictions, valid_targets)
    print("Forests: %.5f" % (hit_rate_forests))
    print valid_prob_forests.shape

     ###############################################################################
    # Adaboost

    num_learner = 600
    ada_clf = AdaBoostClassifier(n_estimators=num_learner,learning_rate=0.2)
    ada_clf = ada_clf.fit(pca.transform(train_data), train_targets)
#     scores = cross_val_score(ada_clf, valid_data_pca, valid_targets)
#     scores.mean()
    valid_prob_adaboost = ada_clf.predict_proba(pca.transform(valid_data))

    valid_predictions = ada_clf.predict(pca.transform(valid_data))
#     print valid_predictions.shape, valid_targets.shape
#     valid_predictions.shape = (num_samples,)
    valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
    hit_rate_adaboost = 1 - percent_error(valid_predictions, valid_targets)
    print("Adaboost: %.5f" % (hit_rate_adaboost))
    print valid_prob_adaboost.shape

    ###############################################################################
    # max probability
    collection_of_prob = []
#     collection_of_prob.append(valid_prob_svm * hit_rate_svm)
#     collection_of_prob.append(valid_prob_forests * hit_rate_forests)
#     collection_of_prob.append(valid_prob_adaboost * hit_rate_adaboost)

    collection_of_prob.append(valid_prob_svm)
    collection_of_prob.append(valid_prob_forests)
    collection_of_prob.append(valid_prob_adaboost)

    predictions = max_proba_predictions(collection_of_prob)
    print predictions.shape, valid_targets.shape
    print "Voted validation hit rate: %.5f" % (1 - percent_error(predictions.reshape(1, predictions.shape[0]), valid_targets))

    ###############################################################################
    print( "try test set")
    test_images = load_data_test("public_test_images").T
    print test_images.shape
    test_data_pca = pca.transform(test_images)

    # svm
    test_prob_svm = svm_clf.predict_proba(test_data_pca)

    # tree
    test_prob_forests = trees_clf.predict_proba(test_images)

    # adaboost
    test_prob_adaboost = ada_clf.predict_proba(test_data_pca)

    # max probability
    collection_of_prob = []
#     collection_of_prob.append(test_prob_svm * hit_rate_svm)
#     collection_of_prob.append(test_prob_forests * hit_rate_forests)
#     collection_of_prob.append(test_prob_adaboost * hit_rate_adaboost)
#
#
    collection_of_prob.append(test_prob_svm)
    collection_of_prob.append(test_prob_forests)
    collection_of_prob.append(test_prob_adaboost)

    test_predictions = max_proba_predictions(collection_of_prob)

    test_predictions = test_predictions.reshape(1, test_predictions.shape[0])
    save_csv(test_predictions)
    print "Done writing CSV file!"

def main_bags_of_svm():
    n = 4
    valid_size = 96  + 96 * n
    train_size= 3000
#
    valid_size = 700
#     train_size= 3000

    num_svm = 3
    SVMs = [svm.SVC(probability=True)] * num_svm
    PCAs = [RandomizedPCA(n_components=80, whiten=True)] * num_svm
    weights = np.ones(num_svm)

    for i in xrange(num_svm):
        filename = 'labeled_images.mat'
        valid_targets, _, valid_data, \
        train_targets, _, train_data = load_data(valid_size, train_size, filename);

        #reshape data and labels
        train_data = train_data.T
        valid_data = valid_data.T
        train_targets = train_targets.T.reshape(train_targets.shape[1])
        valid_targets = valid_targets.T.reshape(valid_targets.shape[1])

        PCAs[i].fit(train_data)

        # transform data
        train_data_pca = PCAs[i].transform(train_data)
        valid_data_pca = PCAs[i].transform(valid_data)
        # reshape targets for fitting


        # can also get probability score
        SVMs[i].fit(train_data_pca, train_targets)
        valid_prob_svm = SVMs[i].predict_proba(valid_data_pca)

        valid_predictions = SVMs[i].predict(valid_data_pca)
        valid_predictions = valid_predictions.reshape(1, valid_predictions.shape[0])
        hit_rate_svm = 1 - percent_error(valid_predictions, valid_targets)
        print("SVM: %.5f" % (hit_rate_svm))
        weights[i] = (hit_rate_svm)

    print "Average validation score: %.5f" % (np.sum(weights)/weights.shape[0])

    # Now used the trained SVM's to vote on a new validation set
    filename = 'labeled_images.mat'
    valid_targets, _, valid_data, \
    _, _, _ = load_data(50, train_size, filename);

    #reshape data and labels
    valid_data = valid_data.T
    valid_targets = valid_targets.T.reshape(valid_targets.shape[1])

    ###############################################################################
    # max probability
    collection_of_prob = []
    for i in xrange(num_svm):
        # transform data
        valid_data_pca = PCAs[i].transform(valid_data)

        # run probabilistic prediction
        valid_prob_svm = SVMs[i].predict_proba(valid_data_pca)
        print valid_prob_svm.shape
        collection_of_prob.append(valid_prob_svm * weights[i])

    predictions = max_proba_predictions(collection_of_prob)
    print (1 - percent_error(predictions.reshape(1, predictions.shape[0]), valid_targets))

    ###############################################################################
    print( "try test set")
    test_images = load_data_test("public_test_images").T
    print test_images.shape

    # max probability
    collection_of_prob = []
    for i in xrange(num_svm):
        # transform data
        test_data_pca = PCAs[i].transform(test_images)

        # run probabilistic prediction
        test_prob_svm = SVMs[i].predict_proba(test_data_pca)
        print test_prob_svm.shape
        collection_of_prob.append(test_prob_svm * weights[i])

    test_predictions = max_proba_predictions(collection_of_prob)

    test_predictions = test_predictions.reshape(1, test_predictions.shape[0])
    save_csv(test_predictions)
    print "Done writing CSV file!"


if __name__ == '__main__':
#     main_single_classifier()
#     main_max_logprob()
    main_bags_of_svm()
