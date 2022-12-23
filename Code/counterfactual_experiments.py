from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
from IPython.display import display
from Utilities.customclf import CustomClf
from Utilities.carla_utilities import determine_feature_types, run_benchmark, make_test_benchmark
from Datasets.optimize import compute_classifier_inputs
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_static_features
from Models.classification import classifier_evaluation
from CARLA.carla.data.catalog import CsvCatalog
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import warnings, os, json, torch
warnings.filterwarnings("ignore")


################################### PATHS ###################################    
if ('Code' in os.getcwd()):
    base_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data', 'Optimise')
else:
    base_path = os.path.join(os.getcwd(), 'Data', 'Optimise')
data_path = base_path + '\\'
save_path = os.path.join(base_path, 'Results', 'Classifiers')

if not os.path.isdir(os.path.join(base_path, 'Results')):
    os.mkdir(os.path.join(base_path, 'Results'))
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
with open(os.path.join(data_path, 'daniel_static_features.json'),'r') as file_object:  
    static_feature_types = json.load(file_object)


################################## READING DATA ###############################


static_features = read_static_features(data_path, static_feature_types)
           
dynamic_features = read_dynamic_features(data_path, visits, assessments)

features, static_feature_types = feature_extraction(dynamic_features, 
                                                    static_features=static_features, 
                                            static_feature_types=static_feature_types, 
                                            aggregation='union')

remissions = compute_output_labels(dynamic_features, labels=criteria)

X, Y, subjects, X_df = compute_classifier_inputs(features, remissions, assessments,
                                            'PANNS', static_feature_types, feature_visits, 
                                            label_visit, 'impute')


############################### CLASSIFICATION ################################
classifiers = [
            # LogisticRegression(),
            # SVC(kernel="linear"),
            # SVC(probability=True),
            # KNeighborsClassifier(),
            # MLPClassifier(),
            # DecisionTreeClassifier(),
            # RandomForestClassifier(),
            AdaBoostClassifier(),
            # GradientBoostingClassifier(),
            # GaussianNB(),
            ]

classifier_names = ['LR', "LSVM", "RBF-SVM", "KNN", "MLP", "DT", "RF", "AdaBoost", 
                    'GB', "NB"]


aucs = classifier_evaluation(features, remissions, cv=10, repetitions=repetitions, assessments=assessments,
                          static_feature_types=static_feature_types, criterion=remission_criterion,
                          feature_visits=feature_visits, label_visit=label_visit, missing_values='impute',
                          classifiers=classifiers, classifier_names=classifier_names)

############################### COUNTERFACTUAL ################################

# Choose best classifier and continue
average_aucs = np.array([(np.mean(aucs[:,i]),np.std(aucs[:,i])) for i in range(len(classifiers))])
best_classifier = classifiers[np.argmax(average_aucs[:,0])]
print(best_classifier)

# Prepare and create carala dataset  
continuous, categorical, immutable = determine_feature_types(X_df, static_feature_types, assessments)
X_df.columns = [str(x) for x in list(X_df.columns)]
X_df['Y'] = Y
dataset = CsvCatalog(df=X_df,
                     continuous=continuous,
                     categorical=categorical,
                     immutables=immutable,
                     target='Y')

# Create custom classifier using carla interfaca
model = CustomClf(dataset, clf=best_classifier, fit_full_data=True)

# predict negative instances to flip later on using one of the counterfactual generation methods
factuals = predict_negative_instances(model, model.X)
test_factual = factuals.iloc[:5]

face_eps_hyperparameters = {'mode':'epsilon', 'fraction': 0.1}
face_knn_hyperparameters = {'mode':'knn', 'fraction': 0.2}
revise_hyperparameters = {
        "data_name": 'test_data',
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.1,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": True,
        "vae_params": {
            "layers": [628, 314, 25],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 20,
            "lr": 1e-3,
            "batch_size": 8,
        }
    }
dice_hyperparamters = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

# init one of the recourse method and make benchmark object (Revise without a descent gpu is very slow)
recourse_method = Revise(model, dataset, revise_hyperparameters)
recourse_method = GrowingSpheres(model)
recourse_method = Face(model, hyperparams=face_eps_hyperparameters)
recourse_method = Face(model, hyperparams=face_knn_hyperparameters)
recourse_method = Dice(model, hyperparams=dice_hyperparamters)

benchmark = Benchmark(model, recourse_method, test_factual)
df_bench, metrics = run_benchmark(benchmark)
display(df_bench)
