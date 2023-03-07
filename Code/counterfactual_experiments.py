from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
from IPython.display import display
from datetime import datetime
from Utilities.customclf import CustomClf
from Plotting.plots import box_plot_benchmark_multiple_rc_methods
from CARLA.carla.plotting.plotting import summary_plot, single_sample_plot
from Utilities.carla_utilities import determine_feature_types, run_benchmark
from Utilities.carla_utilities import generate_batch_counterfactuals_single_factual
from Utilities.carla_utilities import save_all_data_and_parameters
from Utilities.carla_utilities import generate_counterfactuals_for_batch_factuals
from Utilities.carla_utilities import NpEncoder
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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, time
import pandas as pd
import numpy as np
import warnings, os, json
from sklearn.ensemble import BaggingClassifier
warnings.filterwarnings("ignore")


################################### PATHS ###################################    
if ('Code' in os.getcwd()):
    base_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data', 'Optimise')
else:
    base_path = os.path.join(os.getcwd(), 'Data', 'Optimise')
data_path = base_path + '\\'
save_path = os.path.join(base_path, 'Results', 'Classifiers')
carla_save_path = os.path.join(base_path, 'Results', 'Counterfactuals')

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
            # SVC(kernel="linear", probability=True),
            SVC(probability=True),
            # KNeighborsClassifier(),
            # MLPClassifier(),
            # DecisionTreeClassifier(),
            # RandomForestClassifier(),
            # AdaBoostClassifier(),
            # GradientBoostingClassifier(),
            # GaussianNB(),
            ]

classifier_names = ['LR', "LSVM", "RBF-SVM", "KNN", "MLP", "DT", "RF", "AdaBoost", 
                    'GB', "NB"]


aucs = classifier_evaluation(features, remissions, cv=5, repetitions=repetitions, assessments=assessments,
                        static_feature_types=static_feature_types, criterion=remission_criterion,
                        feature_visits=feature_visits, label_visit=label_visit, missing_values='impute',
                        classifiers=classifiers, classifier_names=classifier_names)

############################### COUNTERFACTUAL ################################

# Choose best classifier and continue
average_aucs = np.array([(np.mean(aucs[:,i]),np.std(aucs[:,i])) for i in range(len(classifiers))])
best_classifier = classifiers[np.argmax(average_aucs[:,0])]
best_classifier = BaggingClassifier(best_classifier,n_estimators=10, random_state=0)
print(best_classifier)

# Prepare and create carala dataset  
X_df, continuous, categorical, immutable, translation_dict = determine_feature_types(X_df, static_feature_types, assessments, 
                                                                            use_orginal_column_names=True)
muttable =list(set(continuous+categorical) - set(immutable))
X_df['Y'] = Y
dataset = CsvCatalog(df=X_df,
                    continuous=continuous,
                    categorical=categorical,
                    immutables=immutable,
                    target='Y')

# Create custom classifier using carla interfaca
# TODO: remove fit_full_data param to make pull request possible eventually
model = CustomClf(dataset, clf=best_classifier, fit_full_data=True)

# predict negative instances to flip later on using one of the counterfactual generation methods
factuals = predict_negative_instances(model, model.X)
print(len(factuals))

# read in default hyperparameters for recourse methods
if ('Code' in os.getcwd()):
    hyper_params_path = os.path.join(os.getcwd(), 'Utilities', 'carla_hyper_parameters.json')
else:
    hyper_params_path = os.path.join(os.getcwd(), 'Code', 'Utilities', 'carla_hyper_parameters.json')
hyper_parameters = json.load(open(hyper_params_path))


test_factuals = factuals[:10].copy()

start_time = time.time()
all_results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, test_factuals, 2)   
save_all_data_and_parameters(carla_save_path, all_results, model, hyper_parameters, test_factuals)
time_lapsed = time.time() - start_time
print(time_lapsed)
      
 
#  Read in and plot results
results_folder = os.path.join(carla_save_path, os.listdir(carla_save_path)[2])
rc_methods = ['revise','gs', 'dice','naive_gower', 'cchvae']
rc_results = {}
bench_results = {}
for rc in rc_methods:
    rc_results[rc] = pd.read_csv(os.path.join(results_folder, rc+'_counterfactuals.csv'), index_col=0)
    bench_results[rc] = pd.read_csv(os.path.join(results_folder, rc+'_benchmarks.csv'), index_col=0)
    
box_plot_benchmark_multiple_rc_methods(bench_results, rc_methods, 'L0_distance')
box_plot_benchmark_multiple_rc_methods(bench_results, rc_methods, 'L1_distance')
box_plot_benchmark_multiple_rc_methods(bench_results, rc_methods, 'L2_distance')
box_plot_benchmark_multiple_rc_methods(bench_results, rc_methods, 'single-y-Nearest-Neighbours')
box_plot_benchmark_multiple_rc_methods(bench_results, rc_methods, 'Redundancy')


combined_data = [value for key, value in bench_results.items()]
combined_data = pd.concat(combined_data, keys=rc_methods)
combined_data.index.names = ['method', 'index']
combined_data.reset_index(level='method', inplace=True)
combined_data.reset_index(inplace=True, drop=True)

sns.countplot(data=combined_data, x="method")
sns.barplot(data=combined_data, x="method", y='Redundancy')
pd.crosstab(combined_data['method'],combined_data['connectedness']).plot.barh(stacked=True)
pd.crosstab(combined_data['method'],combined_data['Stability']).plot.barh(stacked=True)


