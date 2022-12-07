#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:19:02 2020

@author: seykia
"""

#################################### IMPORTS ##################################


import warnings
warnings.filterwarnings("ignore")
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_static_features
from Models.classification import classifier_evaluation
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from Plotting.plots import plot_classification_results2
import json
import os
import pickle


################################### PATHS ###################################


# here you shoudld specify the path to OPTIMISE data folder and for saving results
base_path = '/home/preclineu/seykia/Data/Optimize/' 
data_path = base_path + 'Data/'
save_path = base_path + 'Results/Classifiers/'

if not os.path.isdir(base_path + 'Results/'):
    os.mkdir(base_path + 'Results/')
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    

################################ Experiment Configurations ####################
    
# The visits used for the experiment (do not change this)
visits = ['2','3','5']

# The visits used as input to the classifier
feature_visits = ['2']

# The visit used as the output of classifier
label_visit = '5'

# The criterion used for computing the remission labels
remission_criterion = 'PANNS'

# Number of CV repetition
repetitions = 20

# Here specify the assessments for dynamic features
assessments = ['PANNS', 'PSP', 'CGI']

criteria = ['PANNS', 'PSP', 'CGI']

# Here specify the demographics information you want to retrieve
with open(data_path + 'daniel_static_features.json','r') as file_object:  
    static_feature_types = json.load(file_object)


################################## READING DATA ###############################


static_features = read_static_features(data_path, static_feature_types)
           
dynamic_features = read_dynamic_features(data_path, visits, assessments)

features, static_feature_types = feature_extraction(dynamic_features, 
                                                    static_features=static_features, 
                                            static_feature_types=static_feature_types, 
                                            aggregation='union')

remissions =compute_output_labels(dynamic_features, labels=criteria)


############################### CLASSIFICATION ################################


classifiers = [
            LogisticRegression(),
            SVC(kernel="linear"),
            SVC(),
            KNeighborsClassifier(),
            MLPClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            ]

classifier_names = ['LR', "LSVM", "RBF-SVM", "KNN", "MLP", "DT", "RF", "AdaBoost", 
                    'GB', "NB"]


aucs = classifier_evaluation(features, remissions, cv=10, repetitions=repetitions, assessments=assessments,
                          static_feature_types=static_feature_types, criterion=remission_criterion,
                          feature_visits=feature_visits, label_visit=label_visit, missing_values='impute',
                          classifiers=classifiers, classifier_names=classifier_names)



################################## SAVING and PLOTTING ########################


save_file_name = 'Features_' + '_'.join(feature_visits) + '_Label_' + label_visit + \
    '_' + remission_criterion
    
with open(save_path + save_file_name + '.pkl', 'wb') as file:
    pickle.dump({'aucs':aucs, 'classifier_names':classifier_names}, file)
    
    
plot_classification_results2(aucs, classifier_names, save_path=save_path + save_file_name)


###############################################################################
