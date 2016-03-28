#!/usr/bin/python
"""
@Date       : Mar 20, 2016
@Name       : poi_id.py
@Function   :
              * Classify the Enron employees into person's of interest (POI) and
                non-POI
              * Outlier removal
              * Feature selection and engineering
              * Converting features to numpy array
              * Select appropriate classifiers and parameters using Pipelines
              * Evaluate performance measure
@Input      : Financial and Email related data along with labels (POI/non-POI)
@Output     : Evaluation matrix
"""

#############################################################
# import packages, modules, classes
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.grid_search
import sklearn.pipeline
from scipy import interp
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import warnings
warnings.filterwarnings("ignore")
#%%
#############################################################
# Remove outliers
def removeOutliers(data_dict,outliers_list):
    for outlier in outliers_list:
        data_dict.pop(outlier, 0)
    return data_dict

#############################################################
# feature engineering -adding new features
def featureEngineering(data_dict,features):
    for name in data_dict:
        try:
            numerator = data_dict[name]["from_poi_to_this_person"] + data_dict[name]["from_this_person_to_poi"] + data_dict[name]["shared_receipt_with_poi"]
            denominator = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            data_dict[name]['poi_ratio'] = 1.0 * numerator/denominator
        except:
            data_dict[name]['poi_ratio'] = "NaN"

        try:
            data_dict[name]['fraction_from_poi'] = 1.0 * data_dict[name]["from_this_person_to_poi"]/data_dict[name]['from_messages']
        except:
            data_dict[name]['fraction_from_poi'] = "NaN"

        try:
            data_dict[name]['fraction_to_poi'] = 1.0 * data_dict[name]["from_poi_to_this_person"]/data_dict[name]['to_messages']
        except:
            data_dict[name]['fraction_to_poi'] = "NaN"

    return data_dict

#############################################################
# scale features using minmax scaler
def scaleFeatures(features):
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

#############################################################
# apply PCA
def dimensionalityReduction_PCA(clf_list):
    pca = PCA()
    params_pca = {"pca__n_components":[2, 4],
                  "pca__whiten": [False]} #, 4, 5 , 10, 15, 20

    for i in range(len(clf_list)):

        name = "clf_" + str(i)
        clf, params = clf_list[i]

        new_params = {}
        for key, value in params.iteritems():
            new_params[name + "__" + key] = value

        new_params.update(params_pca)
        clf_list[i] = (Pipeline([("pca", pca), (name, clf)]), new_params)

    return clf_list

#############################################################
# Instantiate different classifiers under consideration
def setupSupervisedClassifiers():
    clf_list = []

    # define classifiers and its parameters in tuples

    # Decision Tree
    clf_decisionTree = DecisionTreeClassifier()
    param_decisionTree = {"criterion": ('gini', 'entropy'),
                          "min_samples_split": [2, 4, 6]}
    clf_list.append((clf_decisionTree, param_decisionTree))

    # # Support vector machine
    clf_svc = LinearSVC()
    param_svc = {"C": [0.5, 1, 5, 10, 100, 10**10],
                  "tol":[10**-1, 10**-10],
                  "class_weight":['auto']}
    clf_list.append((clf_svc,param_svc))

     # adaptive boosting - adaboost
    clf_adaboost = AdaBoostClassifier()
    param_adaboost = {"n_estimators":[20, 25, 30, 40, 50, 100]}
    clf_list.append((clf_adaboost,param_adaboost))

     # kNN
    clf_knn = KNeighborsClassifier()
    param_knn = {"n_neighbors":[2, 5], "p":[2,3]}
    clf_list.append((clf_knn,param_knn))

     # Logistic Regression
    clf_logRegression = LogisticRegression()
    param_logRegression = {"C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                            "tol":[10**-1, 10**-5, 10**-10],
                            "class_weight":['auto']}
    clf_list.append((clf_logRegression,param_logRegression))

     # Random RandomForestClassier
    clf_randomForest = RandomForestClassifier( )
    param_randomForest = {"criterion": ('gini', 'entropy')}
    clf_list.append((clf_randomForest,param_randomForest))

    return clf_list

#############################################################
def setupUnSupervisedClassifiers(features_train, labels_train):

    # kMeans
    clf_kmeans = KMeans(n_clusters=2, tol = 0.001)

    pca = PCA(n_components=2, whiten=False)
    clf_kmeans = Pipeline([("pca", pca), ("kmeans", clf_kmeans)])
    clf_kmeans.fit( features_train )

    return [clf_kmeans]

#############################################################
# optimize classifier
def optimize_clf_list(clf_list, features_train, labels_train):

    best_estimators = []
    for clf, params in clf_list:
        clf_optimized = optimize_clf(clf, params, features_train, labels_train)
        best_estimators.append( clf_optimized )

    return best_estimators

#############################################################
# training classifier
def train_clf(features_train, labels_train):

    clf_supervised = setupSupervisedClassifiers()
   # clf_supervised = dimensionalityReduction_PCA(clf_supervised)

    clf_supervised = optimize_clf_list(clf_supervised, features_train, labels_train)
    clf_unsupervised = setupUnSupervisedClassifiers(features_train, labels_train)

    clf_combined = clf_supervised + clf_unsupervised

    return clf_combined

#############################################################
def evaluate_clf(clf, features_test, labels_test):
    labels_pred = clf.predict(features_test)

    f1 = f1_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    accuracy = accuracy_score(labels_test, labels_pred)
    return f1, recall, precision,accuracy

#############################################################
def evaluate_clf_list(clf_list, features_test, labels_test):

    clf_with_scores = []
    for clf in clf_list:
        f1, recall, precision = evaluate_clf(clf, features_test, labels_test)
        clf_with_scores.append( (clf, f1, recall, precision,accuracy) )

    return clf_with_scores

#############################################################
def optimize_clf(clf, params, features_train, labels_train, optimize=True):
    if optimize:
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, params, scoring=scorer)
        clf = clf.fit(features_train, labels_train)
        clf = clf.best_estimator_
    else:
        clf = clf.fit(features_train, labels_train)

    return clf

#############################################################
# Loop through the
def evaluation_loop(features, labels, num_iters=1000, test_size=0.3):
    from numpy import asarray

    evaluation_matrix = [[] for n in range(7)]
    for i in range(num_iters):

        # split data in training and test set
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)

        # Train models
        clf_list = train_clf(features_train, labels_train)

        #loop through all the classifiers
        for i, clf in enumerate(clf_list):
            scores = evaluate_clf(clf, features_test, labels_test)
            evaluation_matrix[i].append(scores)

    summary_list = {}
    for i, col in enumerate(evaluation_matrix):
        summary_list[clf_list[i]] = ( sum(asarray(col)) )

    ordered_list = sorted(summary_list.keys(), key = lambda k: summary_list[k][0], reverse=True)
    return ordered_list, summary_list, evaluation_matrix

#######################################
### Select K best. Makes no sense to use when select k best
### when we are using PCA. (in some cases it might, but here it does not)
def kBestSelector(features, labels, features_list, k = 6):

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)

    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())

    abc = 0
    for k,v in k_best_features.items():
        abc = abc + v

    print " sum >> ", abc
    print k_best_features
    d = k_best_features
    from collections import OrderedDict
    d_descending = OrderedDict(sorted(d.items(), reverse=True))

    features_list = ['poi'] + k_best_features.keys()
    return features_list

features_list_old = features_list
features_list = features_list_old
features_list = kBestSelector(features, labels, features_list, k = 10)
#%%
#############################################################
# Main Program
if __name__ == '__main__':

    from sklearn import cross_validation
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest
    import csv

    features_label      = ["poi"]

    features_email      = [
                           "from_messages",
                           "from_poi_to_this_person",
                           "from_this_person_to_poi",
                           "shared_receipt_with_poi" ,
                           "to_messages"
                           ]

    features_finance    = [
                            "bonus",
                           "deferral_payments",
                           "deferred_income",
                           "director_fees",
                           "exercised_stock_options",
                           "expenses",
                           "loan_advances",
                           "long_term_incentive",
                           "other",
                           "restricted_stock",
                           "restricted_stock_deferred",
                           "salary",
                           "total_payments",
                           "total_stock_value"
                           ]

    features_engineering = ["poi_ratio",
                            "fraction_from_poi",
                            "fraction_to_poi"
                            ]

    # complete features list
    features_list = features_label + features_email + features_finance + features_engineering

    features_list = ["poi", "fraction_from_poi", "fraction_to_poi", 'shared_receipt_with_poi', 'total_stock_value',"deferred_income"]
#    features_list = ['poi', 'long_term_incentive', 'total_stock_value', "fraction_from_poi", "fraction_to_poi", 'shared_receipt_with_poi', 'poi_ratio']

    # Load the pickeled dictionary
    fname = "final_project_dataset.pkl"
    data_dict = pickle.load(open(fname, "r") )

    # outliers list
    outliers_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

    # cleaned data dictionary
    data_dict = removeOutliers(data_dict,outliers_list)

    # feature engineering - adding new features
    data_dict = featureEngineering(data_dict,features_list)

#    features_list = features_list_old
    my_dataset = data_dict

    data = featureFormat(my_dataset, features_list)

    labels, features = targetFeatureSplit(data)

    # Scale features
    features = scaleFeatures(features)

    target_names = ['non-POI', 'POI']

#    features_train, features_test, labels_train, labels_test =  cross_validation.train_test_split(features, labels, test_size = 0.4, random_state = 42)



    # intialize to zero
#    results_acc = np.array([])
#    results_precision = np.array([])
#    results_recall = np.array([])
#    results_f1 = np.array([])

##########################
    kBest = [1, 3, 5, 8]

    # Decision Tree
    pipeline_DT = Pipeline([ # ('sel', SelectKBest()),
                         ('clf', DecisionTreeClassifier())
                         ])

    param_grid_DT = [{ # 'sel__k': kBest,
                   'clf__criterion': ['gini', 'entropy'],
                   'clf__min_samples_split': [2, 4, 5, 6, 10],
                   'clf__max_depth': [2,4,6]
                   }]

##########################
    # SVM
    pipeline_SVM = Pipeline([ # ('sel', SelectKBest()),
                         ('clf', LinearSVC())
                         ])

    param_grid_SVM = [{# 'sel__k': kBest,
                   'clf__C': [0.5, 1, 5, 10, 100, 10**10],
                   'clf__tol': [10**-1, 10**-10]
                   }]

##########################
    # Logistic Regression
    pipeline_LR = Pipeline([#  ('sel', SelectKBest()),
                         ('clf', LogisticRegression())
                         ])

    param_grid_LR = [{ # 'sel__k': kBest,
                   'clf__C': [0.05, 0.5, 1, 10, 10**2,10**5],
                   'clf__tol': [10**-1, 10**-5, 10**-10]
                   }]

##########################
    # adaptive boosting
    pipeline_AdaB = Pipeline([  # ('sel', SelectKBest()),
                         ('clf', AdaBoostClassifier())
                         ])

    param_grid_AdaB = [{ # 'sel__k': kBest,
                   'clf__n_estimators': [1, 3, 5, 10, 25, 30, 40, 50, 100]
                   }]

##########################
    # RandomForestClassifier
    pipeline_RF = Pipeline([ #  ('sel', SelectKBest()),
                         ('clf', RandomForestClassifier())
                         ])

    param_grid_RF = [{ #  'sel__k': kBest,
                   'clf__max_depth': [2, 3, 5],
                    'clf__min_samples_split': [1, 3, 5, 6, 8],
                    'clf__criterion': ['gini', 'entropy']
                   }]

##########################
    # KNN
    pipeline_kNN = Pipeline([  # ('sel', SelectKBest()),
                         ('clf', KNeighborsClassifier())
                         ])

    param_grid_kNN = [{ #  'sel__k': kBest,
                   'clf__n_neighbors': [3, 4, 5, 6, 8],
                    'clf__weights': ['uniform', 'distance']
                   }]

##########################
    # Naive Bayes
    pipeline_NB = Pipeline([ #  ('sel', SelectKBest()),
                         ('clf', GaussianNB())
                         ])

    param_grid_NB = [{ # 'sel__k': kBest
                   }]

##########################
    # Gradient Boost
    pipeline_GB = Pipeline([ #  ('sel', SelectKBest()),
                         ('clf', GradientBoostingClassifier())
                         ])

    param_grid_GB = [{ #  'sel__k': kBest,
                   'clf__max_depth': [2, 3, 5],
                    'clf__n_estimators': [5, 10, 20, 50]
                    }]

##########################

    pipeline_clf = [pipeline_DT, pipeline_SVM, pipeline_LR, pipeline_AdaB, pipeline_RF, pipeline_kNN, pipeline_NB, pipeline_GB]

    pipeline_params = [param_grid_DT, param_grid_SVM, param_grid_LR, param_grid_AdaB, param_grid_RF, param_grid_kNN, param_grid_NB, param_grid_GB]


    clfNames = ['Decision Tree', 'Support Vector Machine', 'Logistic Regression','Adaptive Boost', 'Random Forest', 'k Nearest Neighbor', 'Naive Bayes', 'Gradient Boost']

    lisname = ['Classifier','Precision','Recall','F1','Accuracy']

    start = time.time()

    scoring_index = 'precision' # precision  recall  roc_auc

    # stratified k-fold
    foldsk = 3
    kf = StratifiedKFold(labels,n_folds = foldsk,shuffle=False, random_state=None)

    with open('ClassifierResults.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(lisname)
        for pipeline, param_grid, clfNames in zip(pipeline_clf,pipeline_params,clfNames):
                # intialize to zero
            results_acc = np.array([])
            results_precision = np.array([])
            results_recall = np.array([])
            results_f1 = np.array([])

            for train_indices, test_indices in kf:

                features_train= [features[ii] for ii in train_indices]
                features_test= [features[ii] for ii in test_indices]
                labels_train=[labels[ii] for ii in train_indices]
                labels_test=[labels[ii] for ii in test_indices]

                grid_search = GridSearchCV(pipeline,
                                           param_grid = param_grid,
                                           scoring = scoring_index,
                                           n_jobs=1)

                clf = grid_search.fit(features_train, labels_train)
                pred = clf.best_estimator_.predict(features_test)

                print 'Best Estimator >>> ', grid_search.best_estimator_
                print 'Best score >>> ', grid_search.best_score_
                print 'Best scorer >>> ', grid_search.scorer_
                print 'Best best parameters >>> ', grid_search.best_params_
        #        print 'Best grid scores >>> ', grid_search.grid_scores_

#                report = classification_report(labels_test,pred, target_names=target_names)
        #        print ' best score ', clf.best_score_
    #            print "Classification Report >>> \n", report
    #
    #            print 'accuracy = ', accuracy_score(labels_test,pred)
    #            print 'precision = ', precision_score(labels_test,pred)
    #            print 'recall = ', recall_score(labels_test,pred)
    #            print 'F1 = ', f1_score(labels_test,pred)


                results_acc = np.append(results_acc, [accuracy_score(labels_test,pred)], axis=0)
                results_precision = np.append(results_precision, [precision_score(labels_test,pred)], axis=0)
                results_recall = np.append(results_recall, [recall_score(labels_test,pred)], axis=0)
                results_f1 = np.append(results_f1, [f1_score(labels_test,pred)], axis=0)


            print '>>>>>>>>>>', clfNames
            print "avg precision : ", np.array(results_precision).mean()
            print "avg recall : ", np.array(results_recall).mean()
            print "avg f1 : ", np.array(results_f1).mean()
            print "avg accuracy : ", np.array(results_acc).mean()

            line = []
            line.append(clfNames)
            line.append(np.array(results_precision).mean())
            line.append(np.array(results_recall).mean())
            line.append(np.array(results_f1).mean())
            line.append(np.array(results_acc).mean())
#            print line

            wr.writerow(line)
            print "="*50

    end = time.time()
    print 'Time elapsed = ', (end - start)/60, ' mins'

