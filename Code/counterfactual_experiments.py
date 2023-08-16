from Carla.evaluation import Benchmark
from Carla.models.negative_instances import predict_negative_instances
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import anderson, shapiro, norm
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from Carla.recourse_methods import *
from IPython.display import display
from datetime import datetime
import Carla.evaluation.catalog as evaluation_catalog
from Utilities.customclf import CustomClf
from Plotting.plots import box_plot_benchmark_multiple_rc_methods
from Carla.plotting.plotting import summary_plot, single_sample_plot
from Utilities.carla_utilities import determine_feature_types, run_benchmark, generate_confidence_intervals
from Utilities.carla_utilities import generate_batch_counterfactuals_single_factual
from Utilities.carla_utilities import save_all_data_and_parameters
from Utilities.carla_utilities import generate_counterfactuals_for_batch_factuals
from Utilities.carla_utilities import NpEncoder
from Datasets.optimize import compute_classifier_inputs
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_static_features
from Models.classification import classifier_evaluation
from Carla.data.catalog import CsvCatalog
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
from sklearn.calibration import CalibratedClassifierCV

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

# classifier_names = ['LR', "LSVM", "RBF-SVM", "KNN", "MLP", "DT", "RF", "AdaBoost", 
#                     'GB', "NB"]
classifier_names = ["LR", "RBF-SVM", "KNN", "MLP"]

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
X_df, continuous, categorical, immutable, translation_dict = determine_feature_types(X_df, static_feature_types, assessments, 
                                                                                     use_orginal_column_names=True)
muttable =list(set(continuous+categorical) - set(immutable))
X_df['Y'] = Y
dataset = CsvCatalog(df=X_df,
                    continuous=continuous,
                    categorical=categorical,
                    immutables=immutable,
                    target='Y')
                    # scaling_method='Identity')

# Create custom classifier using carla interface
# TODO: remove fit_full_data param to make pull request possible eventually
model = CustomClf(dataset, clf=best_classifier, scaling=False, fit_full_data=True, calibration='sigmoid', bagging=50)

# predict negative instances to flip later on using one of the counterfactual generation methods
factuals = predict_negative_instances(model, dataset.df[model.X.columns])
print(len(factuals))

# read in default hyperparameters for recourse methods
if ('Code' in os.getcwd()):
    hyper_params_path = os.path.join(os.getcwd(), 'Utilities', 'carla_hyper_parameters.json')
else:
    hyper_params_path = os.path.join(os.getcwd(), 'Code', 'Utilities', 'carla_hyper_parameters.json')
hyper_parameters = json.load(open(hyper_params_path))
hyper_parameters['gower_cf']['retries'] = len(factuals)
hyper_parameters['stability']['ensemble'] = False

test_factuals = factuals.copy()

start_time = time.time()
all_results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, test_factuals[:4], 2, ["gower_cf", "gs"])   
save_all_data_and_parameters(carla_save_path, all_results, model, hyper_parameters, test_factuals)
time_lapsed = time.time() - start_time
print(time_lapsed)

# Example full pipeline
path = os.path.join(carla_save_path, full_runs[2][1])
# benchmarks = pd.read_csv(os.path.join(path, 'gs_benchmarks.csv'), index_col=0)
cfs = pd.read_csv(os.path.join(path, 'gs_counterfactuals.csv'), index_col=0)
loaded_model = pickle.load(open(os.path.join(path, 'model.sav'), 'rb'))

subject = 335
sample = test_factuals.loc[[subject]]
results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, sample, 1, ["gs", "gower_cf"])

sample = test_factuals.loc[[subject]]
cfs = results['gower_cf'][1].copy()
benchmarks = results['gower_cf'][0].copy()
cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
sample["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(sample["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
cfs_probas = generate_confidence_intervals(cfs, loaded_model)
original_probas = generate_confidence_intervals(dataset.df[cfs.columns].loc[cfs.index], loaded_model)
merged_probas = original_probas.join(cfs_probas, lsuffix='_original', rsuffix='_cf')
ssplt = single_sample_plot(sample.loc[subject], cfs.loc[subject], dataset, figsize=(5,2))
print(benchmarks.loc[subject])
print(merged_probas.loc[subject])


sample = test_factuals.loc[[subject]]
cfs = results['gs'][1].copy()
benchmarks = results['gs'][0].copy()
cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
sample["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(sample["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
cfs_probas = pd.DataFrame(loaded_model.predict_proba(cfs), index=cfs.index)
original_probas = pd.DataFrame(loaded_model.predict_proba(dataset.df[cfs.columns]))
cfs_probas = generate_confidence_intervals(cfs_probas, cfs, loaded_model)
original_probas = generate_confidence_intervals(original_probas, dataset.df[cfs.columns], loaded_model)
merged_probas = original_probas.join(cfs_probas, lsuffix='_original', rsuffix='_cf')
ssplt = single_sample_plot(sample.loc[subject], cfs.loc[subject], dataset, figsize=(5,2))
print(benchmarks.loc[subject])
print(merged_probas.loc[subject])


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


# calibration experiments
calib_path = os.path.join(carla_save_path, "10-06-2023T044408_CalibratedClassifierCV(base_estimator=BaggingClassifier(base_estimator=SVC(probability=True),bootstrap=False,n_estimators=50,n_jobs=-1), cv='prefit', n_jobs=-1)_119")
org_path = os.path.join(carla_save_path, '_final_results//30-05-2023T182758_BaggingClassifier(base_estimator=SVC(probability=True), bootstrap=False,n_estimators=100, n_jobs=-1)_89')

org_gs_cfs = pd.read_csv(os.path.join(org_path, 'gs_counterfactuals.csv'), index_col=0)
org_gs_bench = pd.read_csv(os.path.join(org_path, 'gs_benchmarks.csv'), index_col=0)
org_gs_bench['Type'] = 'Original'
org_gs_bench['connectedness'] = org_gs_bench['connectedness'].replace(-1,0)
org_gs_bench.avg_time = org_gs_bench.avg_time.fillna(method='ffill').mean()
org_gs_bench.Success_Rate = org_gs_bench.Success_Rate.fillna(method='ffill').mean()
org_ng_cfs = pd.read_csv(os.path.join(org_path, 'gower_cf_counterfactuals.csv'), index_col=0)
org_ng_bench = pd.read_csv(os.path.join(org_path, 'gower_cf_benchmarks.csv'), index_col=0)
org_ng_bench['Type'] = 'Original'
org_ng_bench['connectedness'] = org_ng_bench['connectedness'].replace(-1,0)
org_ng_bench.avg_time = org_ng_bench.avg_time.fillna(method='ffill').mean()
org_ng_bench.Success_Rate = org_ng_bench.Success_Rate.fillna(method='ffill').mean()
org_model = pickle.load(open(os.path.join(org_path, 'model.sav'), 'rb'))

calib_gs_cfs = pd.read_csv(os.path.join(calib_path, 'gs_counterfactuals.csv'), index_col=0)
calib_gs_bench = pd.read_csv(os.path.join(calib_path, 'gs_benchmarks.csv'), index_col=0)
calib_gs_bench['Type'] = 'Calibrated'
calib_gs_bench['connectedness'] = calib_gs_bench['connectedness'].replace(-1,0)
calib_gs_bench.avg_time = calib_gs_bench.avg_time.fillna(method='ffill').mean()
calib_gs_bench.Success_Rate = calib_gs_bench.Success_Rate.fillna(method='ffill').mean()
calib_ng_cfs = pd.read_csv(os.path.join(calib_path, 'gower_cf_counterfactuals.csv'), index_col=0)
calib_ng_bench = pd.read_csv(os.path.join(calib_path, 'gower_cf_benchmarks.csv'), index_col=0)
calib_ng_bench['Type'] = 'Calibrated'
calib_ng_bench['connectedness'] = calib_ng_bench['connectedness'].replace(-1,0)
calib_ng_bench.avg_time = calib_ng_bench.avg_time.fillna(method='ffill').mean()
calib_ng_bench.Success_Rate = calib_ng_bench.Success_Rate.fillna(method='ffill').mean()
calib_model = pickle.load(open(os.path.join(calib_path, 'model.sav'), 'rb'))

overlap_indices = list(set(org_gs_cfs.index) & set(calib_gs_cfs.index))

cfs_probas_calibrated_gs = calib_model.predict_proba(org_gs_cfs)
cfs_probas_original_gs = org_model.predict_proba(org_gs_cfs)

sns.set(style="white", palette="tab10")
cfs_combined_gs = pd.DataFrame()
cfs_combined_gs['Original'] = cfs_probas_original_gs[:,1:].reshape(1, len(cfs_probas_original_gs))[0]
cfs_combined_gs['Calibrated'] = cfs_probas_calibrated_gs[:,1:].reshape(1, len(cfs_probas_calibrated_gs))[0]
sns.scatterplot(data=cfs_combined_gs)
plt.axhline(y=0.5, color='gray', alpha=0.4) 

cfs_probas_calibrated_ng = calib_model.predict_proba(org_ng_cfs)
cfs_probas_original_ng = org_model.predict_proba(org_ng_cfs)

sns.set(style="white", palette="tab10")
cfs_combined_ng = pd.DataFrame()
cfs_combined_ng['Original'] = cfs_probas_original_ng[:,1:].reshape(1, len(cfs_probas_original_ng))[0]
cfs_combined_ng['Calibrated'] = cfs_probas_calibrated_ng[:,1:].reshape(1, len(cfs_probas_calibrated_ng))[0]
sns.scatterplot(data=cfs_combined_ng)
plt.axhline(y=0.5, color='gray', alpha=0.4) 


combined_data = pd.concat([org_gs_bench, calib_gs_bench])

sns.set(style="white", font_scale=1, palette="ch:start=.2,rot=-.3")
melt_combined = pd.melt(combined_data[['Type', 'single-y-Nearest-Neighbours', 'Stability', 'Redundancy','Sparsity','connectedness', 'Success_Rate']], id_vars=['Type'])
melt_combined.variable.replace('single-y-Nearest-Neighbours', 'single-yNN', inplace=True)
melt_combined.variable.replace('Success_Rate', 'Success Rate', inplace=True)
plt.figure(figsize=(8,4))
ax = sns.barplot(data=melt_combined, x="variable", y='value', hue='Type')
ax.set(xlabel=None)


combined_data = pd.concat([org_ng_bench, calib_ng_bench])

sns.set(style="white", font_scale=1, palette="ch:start=.2,rot=-.3")
melt_combined = pd.melt(combined_data[['Type', 'single-y-Nearest-Neighbours', 'Stability', 'Redundancy','Sparsity','connectedness', 'Success_Rate']], id_vars=['Type'])
melt_combined.variable.replace('single-y-Nearest-Neighbours', 'single-yNN', inplace=True)
melt_combined.variable.replace('Success_Rate', 'Success Rate', inplace=True)
plt.figure(figsize=(8,4))
ax = sns.barplot(data=melt_combined, x="variable", y='value', hue='Type')
ax.set(xlabel=None)



##############################################################################################################################
###############################################   RANDOM STUFF   ############################################################
##############################################################################################################################

#  Read in and plot results
rc_methods = ['gs', 'revise', 'dice','gower_cf', 'cchvae']
full_runs = [('MLP','_final_results//31-05-2023T141114_BaggingClassifier(base_estimator=MLPClassifier(), bootstrap=False,n_estimators=100, n_jobs=-1)_125'), 
              ('LR','_final_results//30-05-2023T152016_BaggingClassifier(base_estimator=LogisticRegression(), bootstrap=False,n_estimators=100, n_jobs=-1)_112'),
              ('SVC','_final_results//30-05-2023T182758_BaggingClassifier(base_estimator=SVC(probability=True), bootstrap=False,n_estimators=100, n_jobs=-1)_89')]
rc_results = {}
bench_results = {}
for run in full_runs:
    results_folder = os.path.join(carla_save_path, run[1])
    clf = run[0]
    bench_results[clf] = {}
    rc_results[clf] = {}
    for rc in rc_methods:
        try:
            rc_results[clf][rc] = pd.read_csv(os.path.join(results_folder, rc+'_counterfactuals.csv'), index_col=0)
            bench_results[clf][rc] = pd.read_csv(os.path.join(results_folder, rc+'_benchmarks.csv'), index_col=0)
        except:
            rc_results[clf][rc] = pd.read_csv(os.path.join(results_folder, 'naive_gower'+'_counterfactuals.csv'), index_col=0)
            bench_results[clf][rc] = pd.read_csv(os.path.join(results_folder, 'naive_gower'+'_benchmarks.csv'), index_col=0)
        if not bench_results[clf][rc].empty:
            bench_results[clf][rc].avg_time = bench_results[clf][rc].avg_time.fillna(method='ffill').mean()
            bench_results[clf][rc].Success_Rate = bench_results[clf][rc].Success_Rate.fillna(method='ffill').mean()
            bench_results[clf][rc]['y-Nearest-Neighbours'] = bench_results[clf][rc]['y-Nearest-Neighbours'].fillna(method='ffill').mean()
            bench_results[clf][rc]['connectedness'] = bench_results[clf][rc]['connectedness'].replace(-1,0)
            bench_results[clf][rc] = bench_results[clf][rc].drop_duplicates()
            
combined_data = []
for clf, value  in bench_results.items():
    for key, val in value.items():
        val.dropna(inplace=True)
    combined_data.append(pd.concat(value, keys=rc_methods))
  
combined_data = pd.concat(combined_data, keys=bench_results.keys())
combined_data.index.names = ['CLF', 'Method', 'index']
combined_data.reset_index(level=['CLF', 'Method'], inplace=True)
combined_data.reset_index(inplace=True, drop=True)

sns.set(style="white", palette="ch:start=.2,rot=-.3")
sns.boxplot(data = combined_data, x = 'L2_distance', y='Method')
    
box_plot_benchmark_multiple_rc_methods(bench_results['MLP'], rc_methods, 'L2_distance')
    
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L2_distance')
box_plot_benchmark_multiple_rc_methods(bench_results['LR'], rc_methods, 'L2_distance')

box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L1_distance')
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'L2_distance')


box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'single-y-Nearest-Neighbours')
box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'single-y-Nearest-Neighbours')

box_plot_benchmark_multiple_rc_methods(bench_results['SVC'], rc_methods, 'Redundancy')



ax = sns.countplot(data=combined_data, x="Method", hue='CLF')
ax = [ax.bar_label(x) for x in ax.containers]

ax = sns.barplot(data=combined_data, x="Method", y='Success_Rate', hue='CLF')
ax = sns.barplot(data=combined_data, x="Method", y='avg_time', hue='CLF')

sns.set(style="white", palette="ch:start=.2,rot=-.3")
fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharey=True)
fig.tight_layout(pad=4)
ax = sns.barplot(ax=axes[0,1], data=combined_data, x="Method", y='Success_Rate', hue='CLF')
ax.legend(loc="upper right", fontsize='17')
# ax.get_legend().remove()
ax.set_xlabel("b) Success rates grouped per classifier", fontsize = 20)
ax.set_ylabel(ylabel= 'Stability', fontsize = 15)
ax.tick_params(labelsize=15)
ax = sns.barplot(ax=axes[0,0], data=combined_data, x="Method", y='Stability', hue='CLF')
ax.get_legend().remove()
ax.set_xlabel("a) Stability ratios grouped per classifier", fontsize = 20)
ax.set_ylabel(ylabel= 'Stability', fontsize = 15)
ax.tick_params(labelsize=15)
ax = sns.barplot(ax=axes[2,1], data=combined_data, x="Method", y='connectedness', hue='CLF')
ax.get_legend().remove()
ax.set(xlabel=None)
ax.tick_params(labelsize=15)
ax.set_xlabel("f) Connectedness ratios grouped per classifier", fontsize = 20)
ax.set_ylabel(ylabel= 'Connectedness', fontsize = 15)
ax = sns.barplot(ax=axes[1,0], data=combined_data, x="Method", y='Redundancy', hue='CLF')
ax.get_legend().remove()
ax.set(xlabel=None)
ax.tick_params(labelsize=15)
ax.set_ylabel(ylabel= 'Redundancy', fontsize = 15)
ax.set_xlabel("c) Redundancy ratios grouped per classifier", fontsize = 20)
ax = sns.barplot(ax=axes[1,1], data=combined_data, x="Method", y='Sparsity', hue='CLF')
ax.get_legend().remove()
ax.set(xlabel=None)
ax.tick_params(labelsize=15)
ax.set_ylabel(ylabel= 'Sparsity', fontsize = 15)
ax.set_xlabel("d) Sparsity ratios grouped per classifier", fontsize = 20)
ax = sns.barplot(ax=axes[2,0], data=combined_data, x="Method", y='single-y-Nearest-Neighbours', hue='CLF')
ax.get_legend().remove()
ax.set(xlabel=None)
ax.tick_params(labelsize=15)
ax.set_ylabel(ylabel= 'Single yNN', fontsize = 15)
ax.set_xlabel("e) Single yNN ratios grouped per classifier", fontsize = 20)
handles, labels = ax.get_legend_handles_labels()
# fig.delaxes(axes[0][1])
# fig.legend(handles, labels, bbox_to_anchor=(0.8, .84), fontsize='18')


sns.set(style="white", palette="ch:start=.2,rot=-.3")
fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharey=True)
ax = sns.barplot(ax=axes[0,0], data=combined_data, x="Method", y='Stability')
ax.set(xlabel="a) Stability ratios grouped per recourse method")
ax.set_ylabel(ylabel= 'Stability', fontsize = 15)
ax.set(xlabel=None)
ax = sns.barplot(ax=axes[2,1], data=combined_data, x="Method", y='connectedness')
ax.set(xlabel="e) Connectedness ratios grouped per recourse method")
ax.set_ylabel(ylabel= 'Connectedness', fontsize = 15)
ax.set(xlabel=None)
ax = sns.barplot(ax=axes[1,0], data=combined_data, x="Method", y='Redundancy')
ax.set(xlabel=None)
ax.set_ylabel(ylabel= 'Redundancy', fontsize = 15)
ax.set(xlabel="b) Redundancy ratios grouped per recourse method")
ax = sns.barplot(ax=axes[1,1], data=combined_data, x="Method", y='Sparsity')
ax.set(xlabel=None)
ax.set_ylabel(ylabel= 'Sparsity', fontsize = 15)
ax.set(xlabel="c) Sparsity ratios grouped per recourse method")
ax = sns.barplot(ax=axes[2,0], data=combined_data, x="Method", y='single-y-Nearest-Neighbours')
ax.set(xlabel=None)
ax.set_ylabel(ylabel= 'Single yNN', fontsize = 15)
ax.set(xlabel="d) Single yNN ratios grouped per recourse method")
handles, labels = ax.get_legend_handles_labels()
fig.delaxes(axes[0][1])
fig.legend(handles, labels, bbox_to_anchor=(0.8, .84), fontsize='18')



ax = sns.barplot(data=combined_data, x="CLF", y='connectedness', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='Stability', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='Redundancy', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='Sparsity', hue='Method')
ax = sns.barplot(data=combined_data, x="CLF", y='single-y-Nearest-Neighbours', hue='Method')

ax = sns.barplot(data=combined_data, x="Method", y='connectedness', hue='CLF')
ax = sns.barplot(data=combined_data, x="Method", y='Stability', hue='CLF')
ax = sns.barplot(data=combined_data, x="Method", y='Redundancy', hue='CLF')
ax = sns.barplot(data=combined_data, x="Method", y='Sparsity', hue='CLF')
ax = sns.barplot(data=combined_data, x="Method", y='single-y-Nearest-Neighbours', hue='CLF')

sns.set(style="white", font_scale=2, palette="ch:start=.2,rot=-.3")
melt_combined = pd.melt(combined_data[['CLF', 'Method', 'single-y-Nearest-Neighbours', 'Stability', 'Redundancy','Sparsity','connectedness']], id_vars=['CLF', 'Method'])
melt_combined.variable.replace('single-y-Nearest-Neighbours', 'single-yNN', inplace=True)
plt.figure(figsize=(20,10))
ax = sns.barplot(data=melt_combined, x="variable", y='value', hue='Method')
ax.set(xlabel=None)
plt.xticks(rotation=90)


sns.set(style="whitegrid", palette="ch:start=.2,rot=-.3")
fig, ax = plt.subplots(1, figsize=(10, 5))
plt.xscale('log')
ax = sns.boxplot(data=combined_data, y="Method", x="L2_distance", orient="h")

fig, ax = plt.subplots(1, figsize=(10, 5))
plt.xscale('log')
ax = sns.boxplot(data=combined_data, y="Method", x="L2_distance", hue='CLF', orient="h")

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

# gower_cf_success_rate analysis
res = {}
for i in [1, 5, 10] + list(range(10,len(test_factuals), 10)):
    hyper_parameters['gower_cf']['retries'] = i
    start_time = time.time()
    all_results = generate_counterfactuals_for_batch_factuals(model, hyper_parameters, test_factuals, 1, ["gower_cf"])   
    # save_all_data_and_parameters(carla_save_path, all_results, model, hyper_parameters, test_factuals)
    time_lapsed = time.time() - start_time
    res[i] = len(all_results['gower_cf'][1])
    print(time_lapsed)


ax = plt.axes()
# ax.set_title('Naive Gower Succes Rate Analysis')
ax.set(xlabel='Threshold', ylabel='Succes Rate %')
plot_data = pd.DataFrame(all_res)
plot_data["Logistic Regression"] = (plot_data["Logistic Regression"] / 125)*100
plot_data["MLP"] = (plot_data["MLP"] / 125)*100
plot_data["SVC"] = (plot_data["SVC"] / 89)*100
sns.lineplot(data=plot_data) 


# random_state experiments
rc_methods = ['gs', 'revise', 'dice','gower_cf', 'cchvae']
connect_folder = os.path.join(carla_save_path, 'connectedness experiments')


# connectedness experiments    
rc_methods = ['gs', 'revise', 'dice','gower_cf', 'cchvae']
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
        
# final_c_bench = {}
# for key, value in cfs_for_conns.items():
#     row = {}
#     # model = CustomClf(dataset, clf=best_classifier, fit_full_data=True)
#     benchmark = Benchmark(mlmodel=model, factuals=None , recourse_method=None, counterfactuals=value)
#     conns = benchmark.run_benchmark([evaluation_catalog.Stability(benchmark.mlmodel, hyper_parameters['stability'])])['Stability']
#     final_c_bench[key] = conns
    
    
final_c_bench = pd.DataFrame()
for eps in np.arange(4.5, 6, 0.5):
    print(eps)
    hyper_parameters['connectedness']['eps'] = eps  
    for min_s in np.arange(30,100,10):
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

sns.set(style="white")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
fig.tight_layout(pad=3.5)

temp_data_min_s = final_c_bench[final_c_bench.min_samples == 4]
axes[0,0].set_title('a) epsilon analysis for min_samples of 4', fontsize=16)
ax = sns.lineplot(ax=axes[0,0], data=temp_data_min_s, x='epsilon', y='Connected%',hue="method")            
# ax.get_legend().remove()

temp_data_min_s = final_c_bench[final_c_bench.min_samples == 6]
axes[0,1].set_title('b) epsilon analysis for min_samples of 6', fontsize=16)
ax = sns.lineplot(ax=axes[0,1], data=temp_data_min_s, x='epsilon', y='Connected%',hue="method")            
ax.get_legend().remove()

temp_data_min_s = final_c_bench[final_c_bench.min_samples == 8]
axes[1,0].set_title('c) epsilon analysis for min_samples of 8', fontsize=16)
ax = sns.lineplot(ax=axes[1,0], data=temp_data_min_s, x='epsilon', y='Connected%',hue="method")            
ax.get_legend().remove()

temp_data_min_s = final_c_bench[final_c_bench.min_samples == 10]
axes[1,1].set_title('d) epsilon analysis for min_samples of 10', fontsize=16)
ax = sns.lineplot(ax=axes[1,1], data=temp_data_min_s, x='epsilon', y='Connected%',hue="method")            
ax.get_legend().remove()






method = 'gs'
temp_data_method = final_c_bench[final_c_bench.method == method]
pivot = temp_data_method[['epsilon', 'Connected%', 'min_samples']].pivot(index='min_samples', columns='epsilon', values='Connected%')
ax = plt.axes()
ax.set_title(method+' recourse')
sns.heatmap(pivot, annot=False, cmap="crest")


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(dataset.df.drop(["Y"], axis=1))
distances, indices = neighbors_fit.kneighbors(dataset.df.drop(["Y"], axis=1))

distances = np.sort(distances, axis=0)
distances = distances[:,1]

ax = plt.axes()
sns.lineplot(distances)
sns.set(style="darkgrid")
ax.set_title('Min Samples')







