import CARLA.carla.evaluation.catalog as evaluation_catalog
from CARLA.carla.data.catalog import OnlineCatalog
from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.catalog import MLModelCatalog
from CARLA.carla.plotting.plotting import summary_plot, single_sample_plot
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
from scipy.stats import anderson, shapiro, norm
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import json, os, pickle, time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval

def make_test_benchmark(data_name="adult", model_name="ann"):
    # get data and mlmodel
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_name, backend="tensorflow")

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[0:5]

    # get recourse method
    hyperparameters = {'mode':'knn', 'fraction': 0.8}
    recourse_method = Face(model)

    # make benchmark object
    benchmark = Benchmark(model, recourse_method, test_factual)

    return benchmark


def run_benchmark(benchmark, hyper_parameters):
    # if benchmark._counterfactuals.dropna().empty:
    #     return pd.DataFrame()
    evaluation_measures = [evaluation_catalog.SingleYNN(benchmark.mlmodel, hyper_parameters['singleYNN']),
                            evaluation_catalog.Stability(benchmark.mlmodel, hyper_parameters['stability']),
                            evaluation_catalog.Redundancy(benchmark.mlmodel, hyper_parameters['redundancy']),
                            evaluation_catalog.Sparsity(benchmark.mlmodel, hyper_parameters['sparsity']),
                            evaluation_catalog.Distance(benchmark.mlmodel),
                            evaluation_catalog.YNN(benchmark.mlmodel, hyper_parameters['ynn']),
                            evaluation_catalog.AvgTime({"time": benchmark.timer}),
                            evaluation_catalog.Connectedness(benchmark.mlmodel, hyper_parameters['connectedness']),
                       evaluation_catalog.SuccessRate()]
    df_benchmark = benchmark.run_benchmark(evaluation_measures)
    return df_benchmark   

def determine_feature_types(X_df, static_feature_types, assessments, use_orginal_column_names=True):
    continuous = []
    categorical = []
    immutable = []
    feature_tanslation_dict = {}
    for feature_group in static_feature_types.keys():
        counter = 0
        for feature in static_feature_types[feature_group].keys():
            immut = False
            if feature_group != 'lifestyle':
                immut = True
            if static_feature_types[feature_group][feature][1] > 2 and static_feature_types[feature_group][feature][0] == 'cat':
                for i in range(static_feature_types[feature_group][feature][1]):
                    if immut:
                        immutable.append(str((feature_group,counter)))
                        feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                    categorical.append(str((feature_group,counter)))
                    feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                    counter += 1
            elif static_feature_types[feature_group][feature][0] == 'con':
                if immut:
                    immutable.append(str((feature_group,counter)))
                    feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                continuous.append(str((feature_group,counter)))
                feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                counter += 1
            else:
                if immut:
                    immutable.append(str((feature_group,counter)))
                    feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                categorical.append(str((feature_group,counter)))
                feature_tanslation_dict[str((feature_group,counter))] = (feature_group, feature, counter)
                counter += 1
    for feature_group in assessments:         
        temp_cols = list(X_df[feature_group].columns)
        asses_cols = [feature_group]*len(temp_cols)
        continuous.extend(list(map(str, list(zip(asses_cols, temp_cols)))))
        immutable.extend(list(map(str, list(zip(asses_cols, temp_cols)))))
        for item in list(map(str, list(zip(asses_cols, temp_cols)))):
            feature_tanslation_dict[item] = literal_eval(item)
            
    if use_orginal_column_names:
        original_column_names = []
        for col in X_df.columns:
            original_column_names.append(feature_tanslation_dict[str(col)])
        for i in range(len(categorical)):
            categorical[i] = str(feature_tanslation_dict[categorical[i]])
        for i in range(len(continuous)):
            continuous[i] = str(feature_tanslation_dict[continuous[i]])
        for i in range(len(immutable)):
            immutable[i] = str(feature_tanslation_dict[immutable[i]])
        X_df.columns = original_column_names
    X_df.columns = [str(x) for x in list(X_df.columns)] 
           
    return X_df, continuous, categorical, immutable, feature_tanslation_dict


def generate_batch_counterfactuals_single_factual(dataset, recourse_method, factual, n, top_n=5, return_plot=True):
    combined_data = []
    for i in range(n):
        print(i)
        counterfactual = recourse_method.get_counterfactuals(factual)
        combined_data.append(counterfactual)
    df = pd.DataFrame(np.array(combined_data)[:,0,:], columns=factual.columns)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        raise Exception("No counterfactuals were found!")
    if return_plot:
        single_plots = []
        for index, value in df.iterrows():
            value.name = factual.iloc[0].name
            single_plots.append(single_sample_plot(factual.iloc[0], value, dataset, figsize=(7,20)))
            plt.close()
        sum_plot = summary_plot(pd.DataFrame(np.repeat(factual.values, n, axis=0), columns=factual.columns), df, dataset, topn=top_n)
        # plt.close()
        plt.title(recourse_method.__class__.__name__)
        sum_plot.show()
        return df, sum_plot, single_plots
    else:
        return df
    
def generate_counterfactuals_for_batch_factuals(model, hyper_parameters, factuals, retries=5,
                                                rc_methods=["dice", "gs", "gower_cf", "revise", "cchvae"]):
    revise_factuals = factuals.copy()
    gs_factuals = factuals.copy()
    dice_factuals = factuals.copy()
    gower_cf_factuals = factuals.copy()
    cchvae_factuals = factuals.copy()
    
    revise_counterfactuals = pd.DataFrame()
    gs_counterfactuals = pd.DataFrame()
    gower_cf_counterfactuals = pd.DataFrame()
    dice_counterfactuals = pd.DataFrame(columns=dice_factuals.columns)
    cchvae_counterfactuals = pd.DataFrame()
    
    revise_benchmarks = pd.DataFrame()
    gs_benchmarks = pd.DataFrame()
    gower_cf_benchmarks = pd.DataFrame()
    dice_benchmarks = pd.DataFrame()
    cchvae_benchmarks = pd.DataFrame()
    
    
    for i in range(retries):       
        # get counterfactuals using all methods
        if gower_cf_factuals.empty == False and "gower_cf" in rc_methods:
            gower_cf_method = GowerCF(model, hyperparams=hyper_parameters['gower_cf'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=gower_cf_factuals, recourse_method=gower_cf_method)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            gower_cf_benchmarks = gower_cf_benchmarks.append(bench_results)
            gower_cf_counterfactuals = gower_cf_counterfactuals.append(cfs)
            # drop all rows because retries is not needed for this method
            gower_cf_factuals.drop(gower_cf_factuals.index, inplace=True)
            print(time_lapsed)
            print('gower_cf', len(gower_cf_counterfactuals))
        if gs_factuals.empty == False and "gs" in rc_methods:
            gs_method = GrowingSpheres(model, hyperparams=hyper_parameters['gs'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=gs_factuals, recourse_method=gs_method)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            gs_benchmarks = gs_benchmarks.append(bench_results)
            gs_counterfactuals = gs_counterfactuals.append(cfs)
            gs_factuals.drop(cfs.index, inplace=True)
            print(time_lapsed)
            print('gs', len(gs_counterfactuals))
        if revise_factuals.empty == False and "revise" in rc_methods:
            revise_method = Revise(model, model.data, hyperparams=hyper_parameters['revise'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=revise_factuals, recourse_method=revise_method)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            revise_benchmarks = revise_benchmarks.append(bench_results)
            revise_counterfactuals = revise_counterfactuals.append(cfs)
            revise_factuals.drop(cfs.index, inplace=True)
            print(time_lapsed)
            print('revise', len(revise_counterfactuals))
        if cchvae_factuals.empty == False and "cchvae" in rc_methods:
            cchvae_method = CCHVAE(model, hyperparams=hyper_parameters['cchvae'])
            benchmark = Benchmark(mlmodel=model, factuals=cchvae_factuals, recourse_method=cchvae_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            cchvae_benchmarks = cchvae_benchmarks.append(bench_results)
            cchvae_counterfactuals = cchvae_counterfactuals.append(cfs)
            cchvae_factuals.drop(cfs.index, inplace=True)
            print('cchvae', len(cchvae_counterfactuals))
        if dice_factuals.empty == False and "dice" in rc_methods:   
            dice_method = Dice(model, hyperparams=hyper_parameters['dice'])
            # for loop only needed for dice due to index disappearance
            for index, value in dice_factuals.iterrows():
                benchmark = Benchmark(mlmodel=model, factuals=dice_factuals.filter(items=[index], axis=0), 
                                      recourse_method=dice_method)
                benchmark._counterfactuals.index = [index]
                dice_cf = benchmark._counterfactuals.copy()
                dice_cf.dropna(inplace=True)
                bench_results = run_benchmark(benchmark, hyper_parameters)
                dice_benchmarks = dice_benchmarks.append(bench_results)
                if dice_cf.empty == False:
                    dice_counterfactuals = dice_counterfactuals.append(dice_cf)
                    dice_factuals.drop(index, inplace=True)
            print('dice', len(dice_counterfactuals))
    all_results = {}
    all_results['revise'] = (revise_benchmarks, revise_counterfactuals)
    all_results['dice'] = (dice_benchmarks, dice_counterfactuals)
    all_results['gs'] = (gs_benchmarks, gs_counterfactuals)
    all_results['cchvae'] = (cchvae_benchmarks, cchvae_counterfactuals)
    all_results['gower_cf'] = (gower_cf_benchmarks, gower_cf_counterfactuals)
    
    return all_results

def single_generate_counterfactuals(model, hyper_parameters, factual, retries=5,
                                                rc_methods=["dice", "gs", "gower_cf", "revise", "cchvae"]):
    factuals = pd.DataFrame(factual.values.reshape(1,len(factual)), columns=factual.index, index=[factual.name])
    revise_factuals = factuals.copy()
    gs_factuals = factuals.copy()
    dice_factuals = factuals.copy()
    gower_cf_factuals = factuals.copy()
    cchvae_factuals = factuals.copy()
    
    revise_counterfactuals = pd.DataFrame()
    gs_counterfactuals = pd.DataFrame()
    gower_cf_counterfactuals = pd.DataFrame()
    dice_counterfactuals = pd.DataFrame(columns=dice_factuals.columns)
    cchvae_counterfactuals = pd.DataFrame()
    
    revise_benchmarks = pd.DataFrame()
    gs_benchmarks = pd.DataFrame()
    gower_cf_benchmarks = pd.DataFrame()
    dice_benchmarks = pd.DataFrame()
    cchvae_benchmarks = pd.DataFrame()
    
    
    for i in range(retries):       
        # get counterfactuals using all methods
        if gower_cf_factuals.empty == False and "gower_cf" in rc_methods:
            gower_cf_method = GowerCF(model, hyperparams=hyper_parameters['gower_cf'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=gower_cf_factuals, recourse_method=gower_cf_method, single_mode=True)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            gower_cf_benchmarks = gower_cf_benchmarks.append(bench_results, ignore_index=True)
            gower_cf_counterfactuals = gower_cf_counterfactuals.append(cfs, ignore_index=True)
            # drop all rows because retries is not needed for this method
            gower_cf_factuals.drop(gower_cf_factuals.index, inplace=True)
            gower_cf_counterfactuals = gower_cf_counterfactuals.iloc[:retries]
            print(time_lapsed)
            print('gower_cf', len(gower_cf_counterfactuals))
        if gs_factuals.empty == False and "gs" in rc_methods:
            gs_method = GrowingSpheres(model, hyperparams=hyper_parameters['gs'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=gs_factuals, recourse_method=gs_method)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            gs_benchmarks = gs_benchmarks.append(bench_results, ignore_index=True)
            gs_counterfactuals = gs_counterfactuals.append(cfs, ignore_index=True)
            print(time_lapsed)
            print('gs', len(gs_counterfactuals))
        if revise_factuals.empty == False and "revise" in rc_methods:
            revise_method = Revise(model, model.data, hyperparams=hyper_parameters['revise'])
            start_time = time.time()
            benchmark = Benchmark(mlmodel=model, factuals=revise_factuals, recourse_method=revise_method)
            time_lapsed = time.time() - start_time
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            revise_benchmarks = revise_benchmarks.append(bench_results, ignore_index=True)
            revise_counterfactuals = revise_counterfactuals.append(cfs, ignore_index=True)
            print(time_lapsed)
            print('revise', len(revise_counterfactuals))
        if cchvae_factuals.empty == False and "cchvae" in rc_methods:
            cchvae_method = CCHVAE(model, hyperparams=hyper_parameters['cchvae'])
            benchmark = Benchmark(mlmodel=model, factuals=cchvae_factuals, recourse_method=cchvae_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            cchvae_benchmarks = cchvae_benchmarks.append(bench_results, ignore_index=True)
            cchvae_counterfactuals = cchvae_counterfactuals.append(cfs, ignore_index=True)
            print('cchvae', len(cchvae_counterfactuals))
        if dice_factuals.empty == False and "dice" in rc_methods:   
            dice_method = Dice(model, hyperparams=hyper_parameters['dice'])
            # for loop only needed for dice due to index disappearance
            for index, value in dice_factuals.iterrows():
                benchmark = Benchmark(mlmodel=model, factuals=dice_factuals.filter(items=[index], axis=0), 
                                      recourse_method=dice_method)
                benchmark._counterfactuals.index = [index]
                dice_cf = benchmark._counterfactuals.copy()
                dice_cf.dropna(inplace=True)
                bench_results = run_benchmark(benchmark, hyper_parameters)
                dice_benchmarks = dice_benchmarks.append(bench_results, ignore_index=True)
                if dice_cf.empty == False:
                    dice_counterfactuals = dice_counterfactuals.append(dice_cf, ignore_index=True)
            print('dice', len(dice_counterfactuals))
    all_results = {}
    all_results['revise'] = (revise_benchmarks, revise_counterfactuals)
    all_results['dice'] = (dice_benchmarks, dice_counterfactuals)
    all_results['gs'] = (gs_benchmarks, gs_counterfactuals)
    all_results['cchvae'] = (cchvae_benchmarks, cchvae_counterfactuals)
    all_results['gower_cf'] = (gower_cf_benchmarks, gower_cf_counterfactuals)
    
    return all_results


def save_all_data_and_parameters(save_path, all_results, clf, hyper_parameters, factuals):
    folder_name = os.path.join(save_path, 
                               datetime.now().strftime("%d-%m-%YT%H%M%S") + '_' + 
                               str(clf.raw_model[0]).replace("\n", "").replace("  ", "") + '_' + str(len(factuals)))
    os.mkdir(folder_name)
    # save model, hyperparamters and all results
    pickle.dump(clf.raw_model, open(os.path.join(folder_name, 'model.sav'), 'wb'))
    json.dump(hyper_parameters, open(os.path.join(folder_name, 'hyper_parameters.json'), 'w'), cls=NpEncoder)        
    for key, value in all_results.items():
        value[0].to_csv(os.path.join(folder_name, f'{key}_benchmarks.csv'))
        value[1].to_csv(os.path.join(folder_name, f'{key}_counterfactuals.csv')) 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

    
def generate_confidence_intervals(df, model, confidence_level = 0.95):
    df_probas = pd.DataFrame(model.predict_proba(df), index=df.index)
    df_probas['1_ci_upper'] = None
    df_probas['1_ci_lower'] = None 
    df_probas['0_ci_upper'] = None
    df_probas['0_ci_lower'] = None
    df_probas['0_error'] = None
    df_probas['1_error'] = None

    base_estimator_probs = []
    for estimator in model['estimator'].estimators_:
        pipe = Pipeline([ ('estimator', estimator)])
        base_estimator_probs.append(pipe.predict_proba(df))

    base_estimator_probs = np.array(base_estimator_probs)
    
    mean_probs = np.mean(base_estimator_probs, axis=0)
    std_probs = np.std(base_estimator_probs, axis=0)
    
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    
    for i in range(len(df)):
        class_intervals = []
        for j in range(mean_probs.shape[1]):
            margin_of_error = z_value * std_probs[i, j]
            df_probas[f'{j}_ci_lower'].iloc[i]  = mean_probs[i, j] - margin_of_error
            df_probas[f'{j}_ci_upper'].iloc[i] = mean_probs[i, j] + margin_of_error
            df_probas[f'{j}_error'] = margin_of_error
    return df_probas

def single_sample_normality_test(cfs, model):
    probs = [est.predict_proba(cfs.iloc[0].values.reshape(1,-1)) for est in model.raw_model['estimator'].estimators_]
    probs = np.array(probs)[:,0,1]
    fig = qqplot(probs, line='s')
    pyplot.show()
    
    result = anderson(probs)

    print('Anderson Statistic: %.3f' % result.statistic)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

    stat, p = shapiro(probs)
    print('Shapiro Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks normal (fail to reject H0)')
    else:
        print('Sample does not look normal (reject H0)')

    return fig

def return_best_cf(all_results, n=1, rank_columns= ['L2_distance', 
                                                    'L1_distance',
                                                    'L0_distance',
                                                    'Redundancy',
                                                    'Sparsity',
                                                    'avg_time',
                                                    'Stability', 
                                                    'single-y-Nearest-Neighbours']):
    rank_down_cols = ['Stability', 'single-y-Nearest-Neighbours']
    rank_up_cols = ['L2_distance', 'L1_distance','L0_distance', 'Redundancy', 'avg_time', 'Sparsity']
    
    combined_bench = pd.DataFrame()
    combined_cfs = pd.DataFrame()
    for key, value in all_results.items():
        temp_df = value[0]
        temp_df['method'] = key
        combined_bench = pd.concat([combined_bench, temp_df], axis=0)
        combined_cfs = pd.concat([combined_cfs, value[1]], axis=0)
    combined_bench.drop(columns=['Success_Rate', 'y-Nearest-Neighbours'], inplace=True)
    combined_bench.avg_time = combined_bench.avg_time.fillna(method='ffill')
    combined_bench.dropna(subset=rank_columns, inplace=True)
    combined_cfs.dropna(inplace=True)
    combined_bench.reset_index(inplace=True, drop=True)
    combined_cfs.reset_index(inplace=True, drop=True)
    combined_cfs.drop_duplicates(inplace=True)
    combined_bench = combined_bench.loc[combined_cfs.index]
    rank_df = combined_bench.copy()
    rank_df.update(combined_bench[list(set(rank_columns) - set(rank_down_cols))]
                   .rank(method='average', ascending=True))
    rank_df.update(combined_bench[list(set(rank_columns) - set(rank_up_cols))]
                   .rank(method='average', ascending=False))

    rank_df['avg_rank'] = rank_df[rank_columns].mean(axis=1)
    combined_bench['avg_rank']  = rank_df['avg_rank']
    
    if combined_bench.loc[rank_df[rank_df.connectedness == 1]
                        .nsmallest(n, 'avg_rank').index].empty == False:
        return (combined_cfs.loc[rank_df[rank_df.connectedness == 1].nsmallest(n, 'avg_rank').index], 
                combined_bench.loc[rank_df[rank_df.connectedness == 1].nsmallest(n, 'avg_rank').index],
                rank_df)
    else:
        print("No connected counterfactuals found, returning best counterfactuals regardless of connectedness.")
        return (combined_cfs.loc[rank_df.nsmallest(n, 'avg_rank').index],
                combined_bench.loc[rank_df.nsmallest(n, 'avg_rank').index],
                rank_df)

def transform_features_to_original_scale(cfs, factuals, subjects, scalers):
    unscaled_factuals = factuals.copy()
    unscaled_cfs = cfs.copy()
    unscaled_cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(unscaled_cfs["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
    unscaled_factuals["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"] = scalers["V2_CAFFEINE_CUPS"].inverse_transform(unscaled_factuals["('lifestyle', 'V2_CAFFEINE_CUPS', 2)"])
    unscaled_cfs.index = subjects
    return unscaled_cfs, unscaled_factuals