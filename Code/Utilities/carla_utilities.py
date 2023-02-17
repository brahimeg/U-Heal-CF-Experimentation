import CARLA.carla.evaluation.catalog as evaluation_catalog
from CARLA.carla.data.catalog import OnlineCatalog
from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.catalog import MLModelCatalog
from CARLA.carla.plotting.plotting import summary_plot, single_sample_plot
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
import numpy as np
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


def run_benchmark(benchmark):
    evaluation_measures = [
        evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
        evaluation_catalog.Distance(benchmark.mlmodel),
        evaluation_catalog.SuccessRate(),
        evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
        # evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
        evaluation_catalog.AvgTime({"time": benchmark.timer}),
    ]
    df_benchmark = benchmark.run_benchmark(evaluation_measures)
    return df_benchmark, evaluation_measures   

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
    
    revise_counterfactuals = pd.DataFrame()
    gs_counterfactuals = pd.DataFrame()
    dice_counterfactuals = pd.DataFrame(columns=dice_factuals.columns)
    
    for i in range(retries):
        # init all recourse methods
        dice_method = Dice(model, hyperparams=hyper_parameters['dice'])
        gs_method = GrowingSpheres(model, hyperparams=hyper_parameters['gs'])
        revise_method = Revise(model, model.data, hyperparams=hyper_parameters['revise'])
        
        # get counterfactuals using all methods
        if gs_factuals.empty == False:
            cfs = gs_method.get_counterfactuals(gs_factuals)
            cfs.dropna(inplace=True)
            gs_counterfactuals = gs_counterfactuals.append(cfs)
            gs_factuals.drop(cfs.index, inplace=True)
            print('gs',len(gs_counterfactuals))
        if revise_factuals.empty == False:
            cfs = revise_method.get_counterfactuals(revise_factuals)
            cfs.dropna(inplace=True)
            revise_counterfactuals = revise_counterfactuals.append(cfs)
            revise_factuals.drop(cfs.index, inplace=True)
            print('rev', len(revise_counterfactuals))
        if dice_factuals.empty == False:    
            # for loop only needed for dice due to index disappearance
            for index, value in dice_factuals.iterrows():
                dice_cf = dice_method.get_counterfactuals(dice_factuals.filter(items=[index], axis=0))
                dice_cf.dropna(inplace=True)
                if dice_cf.empty == False:
                    dice_cf.index = [index]
                    dice_counterfactuals = dice_counterfactuals.append(dice_cf)
                    dice_factuals.drop(index, inplace=True)
            print('dice', len(dice_counterfactuals))
                    
    return revise_counterfactuals, gs_counterfactuals, dice_counterfactuals