#!/usr/bin/python
"""
@filename  	: poi_id.py
"""

#############################################################
# import packages, modules, classes
import nltk
from os import path
from wordcloud import WordCloud
from collections import defaultdict
import matplotlib.pyplot as plt
import time

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan

from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.neural_network import BernoulliRBM

from sklearn.cluster import KMeans
#############################################################
# Remove outliers
def removeOutliers(data_dict,outliers_list):
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    return data_dict

#############################################################
# feature engineering -adding new features
def featureEngineering(data_dict,features):
    for name in data_dict:
        try:
            numerator = data_dict[name]["from_poi_to_this_person"] + data_dict[name]["from_this_person_to_poi"] + data_dict[name]["shared_receipt_with_poi"]
            denominator = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_ratio = 1.0 * numerator/denominator
        except:
            data_dict[name]['poi_ratio'] = 'NaN'
    return data_dict

#############################################################

#############################################################

#############################################################


#############################################################
# Main Program
if __name__ == '__main__':

	# Select appropriate features
	features_label = ["poi"]
	features_email = ["from_messages", "from_poi_to_this_person", "from_this_person_to_poi", "shared_receipt_with_poi", "to_messages"]
	features_finance = ["bonus", "deferral_payments", "deferred_income", "director_fees", "email_address", "exercised_stock_options", "expenses", "loan_advances", "long_term_incentive", "other", "restricted_stock", "restricted_stock_deferred", "salary", "total_payments", "total_stock_value"]
	features_engineering = ["poi_ratio"]

	# complete features list
	features = features_label + features_email + features_finance + features_engineering

    # Load the dictionary
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

    # outliers list - on quick check with the 
    outliers_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

    # cleaned data dictionary
    data_dict = removeOutliers(data_dict,outliers_list)
	
    # feature engineering - adding new features
    data_dict = featureEngineering(data_dict,features)






    # start = time.time()
    # end = time.time()
    # print 'Time elapsed = ', (end - start)/60, ' mins'    