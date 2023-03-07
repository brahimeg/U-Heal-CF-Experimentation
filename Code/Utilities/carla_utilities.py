import CARLA.carla.evaluation.catalog as evaluation_catalog
from CARLA.carla.data.catalog import OnlineCatalog
from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.catalog import MLModelCatalog
from CARLA.carla.plotting.plotting import summary_plot, single_sample_plot
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
import json, os, pickle
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
    if benchmark._counterfactuals.dropna().empty:
        return pd.DataFrame()
    evaluation_measures = [evaluation_catalog.SingleYNN(benchmark.mlmodel, hyper_parameters['singleYNN']),
                            evaluation_catalog.Stability(benchmark.mlmodel, hyper_parameters['stability']),
                            evaluation_catalog.Redundancy(benchmark.mlmodel, hyper_parameters['redundancy']),
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
    
def generate_counterfactuals_for_batch_factuals(model, hyper_parameters, factuals, retries=5):
    revise_factuals = factuals.copy()
    gs_factuals = factuals.copy()
    dice_factuals = factuals.copy()
    naive_gower_factuals = factuals.copy()
    cchvae_factuals = factuals.copy()
    
    revise_counterfactuals = pd.DataFrame()
    gs_counterfactuals = pd.DataFrame()
    naive_gower_counterfactuals = pd.DataFrame()
    dice_counterfactuals = pd.DataFrame(columns=dice_factuals.columns)
    cchvae_counterfactuals = pd.DataFrame()
    
    revise_benchmarks = pd.DataFrame()
    gs_benchmarks = pd.DataFrame()
    naive_gower_benchmarks = pd.DataFrame()
    dice_benchmarks = pd.DataFrame()
    cchvae_benchmarks = pd.DataFrame()
    
    
    for i in range(retries):
        # init all recourse methods
        dice_method = Dice(model, hyperparams=hyper_parameters['dice'])
        gs_method = GrowingSpheres(model, hyperparams=hyper_parameters['gs'])
        naive_gower_method = NaiveGower(model, hyperparams=hyper_parameters['naive_gower'])
        revise_method = Revise(model, model.data, hyperparams=hyper_parameters['revise'])
        cchvae_method = CCHVAE(model, hyperparams=hyper_parameters['cchvae'])
        
        # get counterfactuals using all methods
        if naive_gower_factuals.empty == False:
            benchmark = Benchmark(mlmodel=model, factuals=naive_gower_factuals, recourse_method=naive_gower_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            naive_gower_benchmarks = naive_gower_benchmarks.append(bench_results)
            naive_gower_counterfactuals = naive_gower_counterfactuals.append(cfs)
            # drop all rows because retries is not needed for this method
            naive_gower_factuals.drop(naive_gower_factuals.index, inplace=True)
            print('naive_gower', len(naive_gower_counterfactuals))
        if gs_factuals.empty == False:
            benchmark = Benchmark(mlmodel=model, factuals=gs_factuals, recourse_method=gs_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            gs_benchmarks = gs_benchmarks.append(bench_results)
            gs_counterfactuals = gs_counterfactuals.append(cfs)
            gs_factuals.drop(cfs.index, inplace=True)
            print('gs', len(gs_counterfactuals))
        if revise_factuals.empty == False:
            benchmark = Benchmark(mlmodel=model, factuals=revise_factuals, recourse_method=revise_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            revise_benchmarks = revise_benchmarks.append(bench_results)
            revise_counterfactuals = revise_counterfactuals.append(cfs)
            revise_factuals.drop(cfs.index, inplace=True)
            print('revise', len(revise_counterfactuals))
        if cchvae_factuals.empty == False:
            benchmark = Benchmark(mlmodel=model, factuals=cchvae_factuals, recourse_method=cchvae_method)
            bench_results = run_benchmark(benchmark, hyper_parameters)
            cfs = benchmark._counterfactuals.copy()
            cfs.dropna(inplace=True)
            cchvae_benchmarks = cchvae_benchmarks.append(bench_results)
            cchvae_counterfactuals = cchvae_counterfactuals.append(cfs)
            cchvae_factuals.drop(cfs.index, inplace=True)
            print('cchvae', len(cchvae_counterfactuals))
        if dice_factuals.empty == False:    
            # for loop only needed for dice due to index disappearance
            for index, value in dice_factuals.iterrows():
                benchmark = Benchmark(mlmodel=model, factuals=dice_factuals.filter(items=[index], axis=0), 
                                      recourse_method=dice_method)
                dice_cf = benchmark._counterfactuals.copy()
                dice_cf.dropna(inplace=True)
                if dice_cf.empty == False:
                    bench_results = run_benchmark(benchmark, hyper_parameters)
                    dice_cf.index = [index]
                    dice_counterfactuals = dice_counterfactuals.append(dice_cf)
                    dice_benchmarks = dice_benchmarks.append(bench_results)
                    dice_factuals.drop(index, inplace=True)
            print('dice', len(dice_counterfactuals))
    all_results = {}
    all_results['revise'] = (revise_benchmarks, revise_counterfactuals)
    all_results['dice'] = (dice_benchmarks, dice_counterfactuals)
    all_results['gs'] = (gs_benchmarks, gs_counterfactuals)
    all_results['cchvae'] = (cchvae_benchmarks, cchvae_counterfactuals)
    all_results['naive_gower'] = (naive_gower_benchmarks, naive_gower_counterfactuals)
    
    return all_results


def save_all_data_and_parameters(save_path, all_results, clf, hyper_parameters, factuals):
    folder_name = os.path.join(save_path, 
                            datetime.now().strftime("%d-%m-%YT%H%M%S") + '_' + str(clf.raw_model[1]) 
                            + '_' + str(len(factuals)))
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