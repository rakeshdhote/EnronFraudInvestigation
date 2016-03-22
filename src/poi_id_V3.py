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
import time
import sys
import pickle

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import warnings
warnings.filterwarnings("ignore")
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
    params_pca = {"pca__n_components":[2, 3, 4, 5], 
                  "pca__whiten": [False]} #, 10, 15, 20

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
    param_decisionTree = {"criterion": ('gini', 'entropy')}
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
    clf_supervised = dimensionalityReduction_PCA(clf_supervised)

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
    return f1, recall, precision

#############################################################
def evaluate_clf_list(clf_list, features_test, labels_test):

    clf_with_scores = []
    for clf in clf_list:
        f1, recall, precision = evaluate_clf(clf, features_test, labels_test)
        clf_with_scores.append( (clf, f1, recall, precision) )

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
    return ordered_list, summary_list

#############################################################
# Main Program
if __name__ == '__main__':

	# Select appropriate features
	features_label      = ["poi"]
	features_email      = [
#                           "from_messages", 
#                           "from_poi_to_this_person", 
#                           "from_this_person_to_poi", 
                           "shared_receipt_with_poi" #, 
#                           "to_messages"
                           ]
	features_finance    = [
#                            "bonus", 
#                           "deferral_payments", 
#                           "deferred_income", 
#                           "director_fees", 
                           "exercised_stock_options", 
#                           "expenses", 
#                           "loan_advances", 
#                           "long_term_incentive", 
                           "other", 
                           "restricted_stock", 
#                           "restricted_stock_deferred", 
                           "salary" #, 
#                           "total_payments", 
#                           "total_stock_value"
                           ]
	features_engineering = ["poi_ratio",
                            "fraction_from_poi",
                            "fraction_to_poi"
                            ] 
 #,"total_payments_log","salary_log", "bonus_log", "total_stock_value_log","exercised_stock_options_log",]

	# complete features list
	features_list = features_label + features_email + features_finance + features_engineering

    # Load the pickeled dictionary
	fname = "final_project_dataset.pkl"
	data_dict = pickle.load(open(fname, "r") )

    # outliers list
	outliers_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

    # cleaned data dictionary
	data_dict = removeOutliers(data_dict,outliers_list)
	
    # feature engineering - adding new features
	data_dict = featureEngineering(data_dict,features_list)

    # copy of data_dict and features
	my_dataset = data_dict

    # Convert features to numpy array
	data = featureFormat(my_dataset, features_list, sort_keys = True)

    # Split the dictionary into labels and features
	labels, features = targetFeatureSplit(data)

    # Scale features
	features = scaleFeatures(features)
    
	start = time.time()
 #    # select logistic regression classifier
	# clf_logistic = LogisticRegression(C=10**20, penalty='l2', random_state=42, tol=10**-10, class_weight='auto')

	# pca = PCA(n_components=20)
	# clf = Pipeline(steps=[("pca", pca), ("logistic", clf_logistic)])


	# test_classifier(clf, my_dataset, features_list)
	# dump_classifier_and_data(clf, my_dataset,features_list)
    
    
	ordered_list, summary = evaluation_loop(features, labels, num_iters=500, test_size=0.3)

	print ordered_list
	print "="*50
	print summary
	print "="*50
    
	clf = ordered_list[0]
	scores = summary[clf]
	print "Best classifier is ", clf
	print "With scores of f1, recall, precision: ", scores    
 
	end = time.time()
	print 'Time elapsed = ', (end - start)/60, ' mins'   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    