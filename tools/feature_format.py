#!/usr/bin/python

""" 
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

"""


import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN=True will convert "NaN" string to 0.0
        remove_all_zeroes=True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes=True will omit any data points for which
            any of the features you seek are 0.0
    """


    return_list = []

    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        append = False
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        if remove_all_zeroes:
            all_zeroes = True
            for item in tmp_list:
                if item != 0 and item != "NaN":
                    append = True

        if remove_any_zeroes:
            any_zeroes = False
            if 0 in tmp_list or "NaN" in tmp_list:
                append = False
        if append:
            return_list.append( np.array(tmp_list) )


    return np.array(return_list)


def targetFeatureSplit( data ):

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features




