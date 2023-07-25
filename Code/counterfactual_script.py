from CARLA.carla.models.negative_instances import predict_negative_instances
from Utilities.customclf import CustomClf
from CARLA.carla.plotting.plotting import *
from Utilities.carla_utilities import *
from Datasets.optimize import compute_classifier_inputs
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_static_features
from CARLA.carla.data.catalog import CsvCatalog
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings, os, json
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

features, static_feature_types, scalers = feature_extraction(dynamic_features, 
                                                    static_features=static_features, 
                                                    static_feature_types=static_feature_types, 
                                                    aggregation='union')

remissions = compute_output_labels(dynamic_features, labels=criteria)

X, Y, subjects, X_df = compute_classifier_inputs(features, remissions, assessments,
                                                'PANNS', static_feature_types, feature_visits, 
                                                label_visit, 'impute')


############################### CLASSIFICATION ################################

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

# Create custom classifier using carla interface
# TODO: remove fit_full_data param to make pull request possible eventually
model = CustomClf(dataset, clf=SVC(probability=True), scaling=False, fit_full_data=True, calibration=None, bagging=50)

# predict negative instances to flip later on using one of the counterfactual generation methods
factuals = predict_negative_instances(model, dataset.df[model.X.columns])
print(len(factuals))

# read in default hyperparameters for recourse methods
if ('Code' in os.getcwd()):
    hyper_params_path = os.path.join(os.getcwd(), 'Utilities', 'carla_hyper_parameters.json')
else:
    hyper_params_path = os.path.join(os.getcwd(), 'Code', 'Utilities', 'carla_hyper_parameters.json')
hyper_parameters = json.load(open(hyper_params_path))
hyper_parameters['naive_gower']['retries'] = len(dataset.df) - len(factuals)
hyper_parameters['naive_gower']['single_mode'] = True

# Choose subject from factuals to generate counterfactuals for
subject = factuals.index[0]
all_results = single_generate_counterfactuals(model, hyper_parameters, factuals.loc[subject],
                                                          3, ["naive_gower","gs", "revise"])   

# rank generated counterfactuals using average rank method across different metrics
cf, bench = return_best_cf(all_results)
cf_unscaled, factual_unscaled = transform_features_to_original_scale(cf, factuals, [subject], scalers)
bench.index = [subject]
cf.index = [subject]
# do normality test using the bagged classifier and generate CI intervals for the best ranked counterfactual
single_sample_normality_test(cf, model)
cf_probas = generate_confidence_intervals(cf, model.raw_model)
original_probas = generate_confidence_intervals(factuals.loc[[subject]], model.raw_model)
merged_probas = original_probas.join(cf_probas, lsuffix='_original', rsuffix='_cf')
diff_vals, ssplt = single_sample_plot(factual_unscaled.loc[subject], cf_unscaled.loc[subject], dataset, figsize=(5,2))
print(diff_vals)
print(bench.loc[subject])
print(merged_probas.loc[subject])
