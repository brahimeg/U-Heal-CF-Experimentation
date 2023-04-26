from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.negative_instances import predict_negative_instances
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import anderson, shapiro, norm
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from CARLA.carla.recourse_methods import *
from IPython.display import display
from datetime import datetime
import CARLA.carla.evaluation.catalog as evaluation_catalog
from Utilities.customclf import CustomClf
from Plotting.plots import box_plot_benchmark_multiple_rc_methods
from CARLA.carla.plotting.plotting import summary_plot, single_sample_plot
from Utilities.carla_utilities import determine_feature_types, run_benchmark, generate_confidence_intervals
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
from sklearn.ensemble import BaggingClassifier, VotingClassifier
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


aucs = classifier_evaluation(features, remissions, cv=10, repetitions=repetitions, assessments=assessments,
                        static_feature_types=static_feature_types, criterion=remission_criterion,
                        feature_visits=feature_visits, label_visit=label_visit, missing_values='impute',
                        classifiers=classifiers, classifier_names=classifier_names)

############################### COUNTERFACTUAL ################################

# Choose best classifier and continue
average_aucs = np.array([(np.mean(aucs[:,i]),np.std(aucs[:,i])) for i in range(len(classifiers))])
best_classifier = classifiers[np.argmax(average_aucs[:,0])]
best_classifier = BaggingClassifier(best_classifier, n_estimators=5, bootstrap=False, n_jobs=-1)
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
                    target='Y',
                    scaling_method='MinMax')

# Create custom classifier using carla interface
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
hyper_parameters['naive_gower']['retries'] = len(factuals)
# hyper_parameters['stability']['ensemble'] = False

test_factuals = factuals.copy()

start_time = time.time()
all_results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, test_factuals, 1, ["naive_gower"])   
save_all_data_and_parameters(carla_save_path, all_results, model, hyper_parameters, test_factuals)
time_lapsed = time.time() - start_time
print(time_lapsed)

# Example full pipeline
subject = 23
path = results_folder = os.path.join(carla_save_path, 
                                     '29-03-2023T013022_BaggingClassifier(base_estimator=MLPClassifier(), bootstrap=False,n_estimators=30, n_jobs=-1)_125')
benchmarks = pd.read_csv(os.path.join(path, 'gs_benchmarks.csv'), index_col=0)
cfs = pd.read_csv(os.path.join(path, 'gs_counterfactuals.csv'), index_col=0)
loaded_model = pickle.load(open(os.path.join(path, 'model.sav'), 'rb'))

cfs_probas = pd.DataFrame(loaded_model.predict_proba(cfs), index=cfs.index)
original_probas = pd.DataFrame(loaded_model.predict_proba(dataset.df[cfs.columns]))

cfs_probas = generate_confidence_intervals(cfs_probas, cfs, loaded_model)
original_probas = generate_confidence_intervals(original_probas, dataset.df[cfs.columns], loaded_model)

merged_probas = original_probas.join(cfs_probas, lsuffix='_original', rsuffix='_cf')

single_sample_plot(dataset.df.loc[subject], cfs.loc[subject], dataset, figsize=(5,2))


a = dataset.df[cfs.columns].loc[subject]
b = dataset.scaler.inverse_transform([a])
aa = loaded_model['transformer'].inverse_transform([a])

dataset.scaler

# Normality tests for single subject
probs = [est.predict_proba(cfs.loc[subject].values.reshape(1,-1)) for est in loaded_model['estimator'].estimators_]
probs = np.array(probs)[:,0,1]
qqplot(probs, line='s')
pyplot.show()

result = anderson(probs)

print('Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
 sl, cv = result.significance_level[i], result.critical_values[i]
 if result.statistic < result.critical_values[i]:
     print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
 else:
     print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

stat, p = shapiro(probs)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


##############################################################################################################################
###############################################   RANDOM STUFF   ############################################################
##############################################################################################################################

#  Read in and plot results
rc_methods = ['gs', 'revise', 'dice','naive_gower', 'cchvae']
full_runs = [('MLP','29-03-2023T013022_BaggingClassifier(base_estimator=MLPClassifier(), bootstrap=False,n_estimators=30, n_jobs=-1)_125'), 
              ('GB','28-03-2023T201728_BaggingClassifier(base_estimator=GradientBoostingClassifier(), bootstrap=False,n_estimators=30, n_jobs=-1)_123'),
              ('SVC','29-03-2023T045353_BaggingClassifier(base_estimator=SVC(probability=True), bootstrap=False,n_estimators=30, n_jobs=-1)_108')]
rc_results = {}
bench_results = {}
for run in full_runs:
    results_folder = os.path.join(carla_save_path, run[1])
    clf = run[0]
    bench_results[clf] = {}
    rc_results[clf] = {}
    for rc in rc_methods:
        rc_results[clf][rc] = pd.read_csv(os.path.join(results_folder, rc+'_counterfactuals.csv'), index_col=0)
        bench_results[clf][rc] = pd.read_csv(os.path.join(results_folder, rc+'_benchmarks.csv'), index_col=0)
        if not bench_results[clf][rc].empty:
            bench_results[clf][rc].avg_time = bench_results[clf][rc].avg_time.fillna(method='ffill').mean()
            bench_results[clf][rc].Success_Rate = bench_results[clf][rc].Success_Rate.fillna(method='ffill').mean()
            bench_results[clf][rc]['y-Nearest-Neighbours'] = bench_results[clf][rc]['y-Nearest-Neighbours'].fillna(method='ffill').mean()
            bench_results[clf][rc]['connectedness'] = bench_results[clf][rc]['connectedness'].replace(-1,0)
            bench_results[clf][rc] = bench_results[clf][rc].drop_duplicates()
    
box_plot_benchmark_multiple_rc_methods(bench_results['MLP'], rc_methods, 'L2_distance')
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L2_distance')


box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L1_distance')
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L2_distance')


box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'single-y-Nearest-Neighbours')
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'single-y-Nearest-Neighbours')

box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'Redundancy')


combined_data = []
for clf, value  in bench_results.items():
    for key, val in value.items():
        val.dropna(inplace=True)
    combined_data.append(pd.concat(value, keys=rc_methods))
  
combined_data = pd.concat(combined_data, keys=bench_results.keys())
combined_data.index.names = ['CLF', 'Method', 'index']
combined_data.reset_index(level=['CLF', 'Method'], inplace=True)
combined_data.reset_index(inplace=True, drop=True)

ax = sns.countplot(data=combined_data, x="Method", hue='CLF')
ax = [ax.bar_label(x) for x in ax.containers]

sns.barplot(data=combined_data, x="Method", y='Success_Rate', hue='CLF')
sns.barplot(data=combined_data, x="Method", y='avg_time', hue='CLF')
ax = sns.barplot(data=combined_data, x="CLF", y='Stability', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='connectedness', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='Redundancy', hue='Method')
ax = sns.barplot(data=combined_data, x="Method", y='single-y-Nearest-Neighbours', hue='CLF')

sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

pd.crosstab(combined_data['Method'], combined_data['connectedness']).plot.barh(stacked=True)
pd.crosstab(combined_data['Method'], combined_data['Stability']).plot.barh(stacked=True)


# single clf comparison plot
sns.set(style="darkgrid")
method = 'gs'
combined_clf_data = pd.DataFrame()
for key, value in bench_results.items():
    temp = bench_results[key][method]
    temp['clf'] = key
    combined_clf_data = combined_clf_data.append(temp)
    
ax = plt.axes()
ax.set_title(method.upper())
ax = sns.boxplot(data = combined_clf_data, x = 'L2_distance', y='clf')
ax = sns.stripplot(data = combined_clf_data, x = 'L2_distance', y='clf',
                color = 'black',
                alpha = 0.3)

# naive_gower_success_rate analysis
res = {}
for i in [1, 5, 10] + list(range(10,len(test_factuals), 10)):
    hyper_parameters['naive_gower']['retries'] = i
    start_time = time.time()
    all_results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, test_factuals, 1, ["naive_gower"])   
    save_all_data_and_parameters(carla_save_path, all_results, model, hyper_parameters, test_factuals)
    time_lapsed = time.time() - start_time
    res[i] = len(all_results['naive_gower'][1])
    print(time_lapsed)


ax = plt.axes()
ax.set_title('Naive Gower Succes Rate Analysis (SVC)')
plot_data = pd.DataFrame(columns = ['threshold count', 'succes rate %'])
plot_data['threshold count'] = list(res.keys())
plot_data['succes rate %'] = [x/len(factuals) for x in list(res.values())]
sns.lineplot(data=plot_data, x='threshold count', y='succes rate %') 

# random_state experiments
rc_methods = ['gs', 'revise', 'dice','naive_gower', 'cchvae']
connect_folder = os.path.join(carla_save_path, 'connectedness experiments')


# connectedness experiments    
rc_methods = ['gs', 'revise', 'dice','naive_gower', 'cchvae']
connect_folder = os.path.join(carla_save_path, 'connectedness experiments')
cfs_for_conns = {}
models = {}
existing_indc = list(set(X_df.index) - set(factuals.index))
for method in rc_methods:
    cfs_for_conns[method] = pd.DataFrame()
    for folder in os.listdir(connect_folder)[1:]:
        folder_path = os.path.join(connect_folder, folder)
        try:
            temp_df = pd.read_csv(os.path.join(folder_path, method+'_counterfactuals.csv'), index_col=0)
            cfs_for_conns[method] = cfs_for_conns[method].append(temp_df)
            cfs_for_conns[method].drop_duplicates(inplace=True)
            # cfs_for_conns[method].drop(existing_indc, errors='ignore', inplace=True)
            models[method] = pickle.load(open(os.path.join(folder_path,'model.sav'), 'rb')) 
        except:
            print(method)
            continue
        
        
final_c_bench = {}
for key, value in cfs_for_conns.items():
    row = {}
    # model = CustomClf(dataset, clf=best_classifier, fit_full_data=True)
    benchmark = Benchmark(mlmodel=model, factuals=None , recourse_method=None, counterfactuals=value)
    conns = benchmark.run_benchmark([evaluation_catalog.Stability(benchmark.mlmodel, hyper_parameters['stability'])])['Stability']
    final_c_bench[key] = conns
    
    

final_c_bench = pd.DataFrame()
for eps in np.arange(3.0, 4.5, 0.5):
    print(eps)
    hyper_parameters['connectedness']['eps'] = eps  
    for min_s in np.arange(3,10,1):
        hyper_parameters['connectedness']['min_samples'] = min_s  
        for key, value in cfs_for_conns.items():
            row = {}
            benchmark = Benchmark(mlmodel=model, factuals=None , recourse_method=None, counterfactuals=value)
            conns = benchmark.run_benchmark([evaluation_catalog.Connectedness(benchmark.mlmodel, hyper_parameters['connectedness'])])['connectedness']
            row['epsilon'] = eps
            row['min_samples'] = min_s
            row['method'] = key
            try:
                row['Connected%'] = (conns.value_counts()[1] / len(conns))*100
            except:
                row['Connected%'] = 0
            final_c_bench = final_c_bench.append(row, ignore_index=True)

min_s = 9
ax = plt.axes()
ax.set_title('Min Samples ' + str(min_s))
temp_data_min_s = final_c_bench[final_c_bench.min_samples == min_s]
sns.lineplot(data=temp_data_min_s, x='epsilon', y='Connected%',hue="method")            

method = 'revise'
temp_data_method = final_c_bench[final_c_bench.method == method]
pivot = temp_data_method[['epsilon', 'Connected%', 'min_samples']].pivot(index='min_samples', columns='epsilon', values='Connected%')
ax = plt.axes()
ax.set_title(method+' recourse')
sns.heatmap(pivot, annot=False, cmap="crest")

