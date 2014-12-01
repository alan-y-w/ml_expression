import numpy as np
from util import *
from sklearn import svm, grid_search
from sklearn.decomposition import RandomizedPCA, FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import knn

""" ########### WRAPPER FUNCTIONS ########### """

"""
PCA Method.
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_ica(train_data, test_data, ica_count=80):
    ###############################################################################
    # ICA
    pca = PCA(n_components=ica_count, whiten=True).fit(train_data)
    train_data_pca = pca.transform(train_data)
    valid_data_pca = pca.transform(test_data)

    ica = FastICA(n_components=ica_count, max_iter=800, whiten=False).fit(train_data_pca)
    train_data_ica = ica.transform(train_data_pca)
    valid_data_ica = ica.transform(valid_data_pca)
    return train_data_ica, valid_data_ica, ica

"""
ICA Method.
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_pca(train_data, test_data, ica_count=80):
    ###############################################################################
    # ICA
    pca = RandomizedPCA(n_components=ica_count, whiten=True).fit(train_data)
    train_data_pca = pca.transform(train_data)
    valid_data_pca = pca.transform(test_data)
    return train_data_pca, valid_data_pca, pca


"""
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_svm(train_data, train_targets, test_data, test_targets):
    # Search for the best parameter.

    # svr = svm.SVC()
    # parameters = {'kernel':('poly', 'rbf'), 'C':[1, 5, 10], 'gamma':[0.0,0.1,0.2]}
    # svm_clf = grid_search.GridSearchCV(svr, parameters)

    svm_clf = svm.SVC(C=1.0, kernel='rbf', gamma=0, probability=False)
    # can also get probability score
    svm_clf.fit(train_data, train_targets)
    valid_predictions = svm_clf.predict(test_data)
    return svm_clf, (1 - percent_error(valid_predictions, test_targets)), valid_predictions



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

    return trees_clf, (1 - percent_error(valid_predictions, test_targets)), valid_predictions

"""
input:
    data: (#samples, features)
    targets: (#samples,)
"""
def wrapper_adaboost(train_data, train_targets, test_data, test_targets, num_learner=50):

    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=num_learner, learning_rate=0.5)
    ada_clf = ada_clf.fit(train_data, train_targets)
#     scores = cross_val_score(ada_clf, valid_data_pca, valid_targets)
#     scores.mean()
    test_errors = []
    for discrete_train_predict in zip(ada_clf.staged_predict(test_data)):
        test_errors.append(percent_error(discrete_train_predict[0], test_targets.T))
    valid_predictions = ada_clf.predict(test_data)
#     print valid_predictions.shape, valid_targets.shape
#     valid_predictions.shape = (num_samples,)
#     plt.figure(figsize=(5, 5))
#     n_trees_discrete = len(ada_clf)
#     plt.subplot(131)
#     plt.plot(range(1, n_trees_discrete + 1),
#              test_errors, c='black', label='SAMME')
#     plt.legend()
#     plt.ylim(0.18, 0.62)
#     plt.ylabel('Test Error')
#     plt.xlabel('Number of Trees')
#     plt.show()
    return ada_clf, 1 - percent_error(valid_predictions, test_targets), valid_predictions

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
    train_size = 2830 - 96 * n
    filename = 'labeled_images.mat'
    valid_targets, valid_ids, valid_data, \
    train_targets, train_ids, train_data = load_data(valid_size, train_size, filename);
    # print valid_targets.shape, valid_ids.shape, valid_data.shape
    # print train_targets.shape, train_ids.shape, train_data.shape

    ###############################################################################
    # PCA-SVM

    #90 components gives the best result with SVM
    PCA = [60, 70, 80, 85, 90, 95, 100, 110]
    repeat = 10
    best_hit = 0
    for i in PCA:
        tot_hit_rate = 0.
        pca_train, pca_validate, pca = wrapper_pca(train_data, valid_data, i)
        for k in xrange(repeat):
            svm, hit_rate_svm, prediction_svm = wrapper_svm(pca_train, train_targets, pca_validate, valid_targets)
            tot_hit_rate += hit_rate_svm
        avg_hit = tot_hit_rate/repeat
        print ("PCA-SVM hit rate for %d: %.5f" %(i, avg_hit))
        if avg_hit >= best_hit:
            best_prediction_svm = prediction_svm
            best_hit = avg_hit
            best_i = i
            best_pca = pca
            best_pca_train_data = pca_train
            best_pca_validate_data = pca_validate
    print "the best hit is when PCA component is %d, the accuracy is %.5f"%(best_i, best_hit)
        # print ("PCA-SVM hit rate with processing for %d: %.5f" %(i, tot_p_hit_rate/repeat))

    ###############################################################################
    # ICA-SVM

    #90 components gives the best result with SVM
    ICA = [60, 70, 80, 90, 100, 110]
    repeat = 10
    best_hit = 0
    for i in ICA:
        tot_hit_rate = 0.
        ica_train, ica_validate, ica = wrapper_ica(train_data, valid_data, i)
        for k in xrange(repeat):
            svm, hit_rate_svm, prediction_svm = wrapper_svm(ica_train, train_targets, ica_validate, valid_targets)
            tot_hit_rate += hit_rate_svm
        avg_hit = tot_hit_rate/repeat
        print ("ICA-SVM hit rate for %d: %.5f" %(i, avg_hit))
        if avg_hit >= best_hit:
            best_prediction_svm = prediction_svm
            best_hit = avg_hit
            best_i = i
            best_ica = ica
            best_ica_train_data = ica_train
            best_ica_validate_data = ica_validate
    print "the best hit is when ICA component is %d, the accuracy is %.5f"%(best_i, best_hit)

    ###############################################################################
    # Random forest

    # 200 trees give the best result
    num_tree = [80, 100, 120, 140, 180, 200, 220, 240]
    repeat = 10
    best_hit = 0
    for i in num_tree:
        tot_hit_rate = 0.
        for k in xrange(repeat):
            ada, hit_rate_rft, prediction_rft = wrapper_random_forest(best_pca_train_data, train_targets,\
                                            best_pca_validate_data, valid_targets, i)
            tot_hit_rate += hit_rate_rft
        avg_hit = tot_hit_rate/repeat
        print ("random forest hit rate: %.5f" % avg_hit)
        if avg_hit >= best_hit:
            best_prediction_rft = prediction_rft
            best_hit = avg_hit
            best_i = i
    print "the best hit is when the number of random tree is %d, the accuracy is %.5f"%(best_i, best_hit)
    compare_classification_err(best_prediction_svm, best_prediction_rft, valid_targets)

    ###############################################################################
    # Adaboost

    num_learner = [350, 400, 450, 500, 550, 600, 650, 700]
    repeat = 20
    best_hit = 0
    for i in num_learner:
        tot_hit_rate = 0.
        for k in xrange(repeat):
            ada, hit_rate_ada, prediction_ada = wrapper_adaboost(best_pca_train_data, train_targets,\
                                            best_pca_validate_data, valid_targets, i)
            tot_hit_rate += hit_rate_ada
        avg_hit = tot_hit_rate/repeat
        print ("adaboosted pca-svm hit rate: %.5f" % avg_hit)
        if avg_hit >= best_hit:
            best_prediction_ada = prediction_ada
            best_hit = avg_hit
            best_i = i
    print "the best hit is when the number of aadaboost learner is %d, the accuracy is %.5f"%(best_i, best_hit)
    compare_classification_err(best_prediction_svm, best_prediction_ada, valid_targets)

    ###############################################################################
    # KNN
    KNN = [25,27,30,33,35]
    repeat = 20
    best_hit = 0
    for K in KNN:
        tot_hit_rate = 0.

        for k in xrange(repeat):
            prediction_knn = knn.run_knn(K, best_pca_train_data.T, train_targets, best_pca_validate_data.T)

            tot_hit_rate += 1 - percent_error(prediction_knn, valid_targets)
            avg_hit = tot_hit_rate/repeat

        print "KNN hit rate for %d KNN: %.5f" % (K, avg_hit)

        if avg_hit >= best_hit:
            best_prediction_knn = prediction_knn
            best_hit = avg_hit
            best_i = i
    print "the best hit is when the number of KNN is %d, the accuracy is %.5f"%(best_i, best_hit)
    compare_classification_err(best_prediction_svm, best_prediction_knn, valid_targets)

    ###############################################################################
    # Extra trees
    n = [40, 60, 80, 100, 120, 140, 180, 200]
    for num in n:
        tot_hit_rate = 0.
        for k in xrange(repeat):
            extraTrees = ExtraTreesClassifier(n_estimators=num)
            extraTrees.fit(best_pca_train_data, train_targets)
            prediction_etree = extraTrees.predict(best_pca_validate_data)

            tot_hit_rate += 1 - percent_error(prediction_etree, valid_targets)
            avg_hit = tot_hit_rate/repeat

        print "extraTrees hit rate for %d: %.5f" % (num, avg_hit)

        if avg_hit >= best_hit:
            best_prediction_etree = prediction_etree
            best_hit = avg_hit
            best_i = i
    print "the best hit is when the number of extraTrees is %d, the accuracy is %.5f"%(best_i, best_hit)
    compare_classification_err(best_prediction_svm, best_prediction_etree, valid_targets)

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
    print "Voted validation hit rate: %.5f" % (1 - percent_error(predictions, valid_targets))

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
    valid_size = 100
#     train_size= 3000

    num_svm = 10
    SVMs = [svm.SVC(probability=True)] * num_svm
    PCAs = [RandomizedPCA(n_components=90, whiten=True)] * num_svm
    weights = np.ones(num_svm)

    for i in xrange(num_svm):
        filename = 'labeled_images.mat'
        valid_targets, _, valid_data, \
        train_targets, _, train_data = load_data(valid_size, train_size, filename)

        PCAs[i].fit(train_data)

        # transform data
        train_data_pca = PCAs[i].transform(train_data)
        valid_data_pca = PCAs[i].transform(valid_data)

        # can also get probability score
        SVMs[i].fit(train_data_pca, train_targets)
        valid_prob_svm = SVMs[i].predict_proba(valid_data_pca)

        valid_predictions = SVMs[i].predict(valid_data_pca)
        hit_rate_svm = 1 - percent_error(valid_predictions, valid_targets)
        print("SVM: %.5f" % (hit_rate_svm))
        weights[i] = (hit_rate_svm)




    print "Average validation score: %.5f" % (np.sum(weights)/weights.shape[0])

    # Now used the trained SVM's to vote on a new validation set
    filename = 'labeled_images.mat'
    valid_targets, _, valid_data, \
    _, _, _ = load_data(50, train_size, filename);


    ###############################################################################
    # max probability
    collection_of_prob = []
    for i in xrange(num_svm):
        # transform data
        valid_data_pca = PCAs[i].transform(valid_data)

        # run probabilistic prediction
        valid_prob_svm = SVMs[i].predict_proba(valid_data_pca)
        collection_of_prob.append(valid_prob_svm * weights[i])

    predictions = max_proba_predictions(collection_of_prob)
    print (1 - percent_error(predictions, valid_targets))

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
     main_single_classifier()
    # main_max_logprob()
    # main_bags_of_svm()
