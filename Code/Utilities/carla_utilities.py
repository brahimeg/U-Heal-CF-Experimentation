import CARLA.carla.evaluation.catalog as evaluation_catalog
from CARLA.carla.data.catalog import OnlineCatalog
from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.catalog import MLModelCatalog
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *

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
        evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
        evaluation_catalog.AvgTime({"time": benchmark.timer}),
    ]
    df_benchmark = benchmark.run_benchmark(evaluation_measures)
    return df_benchmark, evaluation_measures   

def determine_feature_types(X_df, static_feature_types, assessments):
    continuous = []
    categorical = []
    immutable = []
    for feature_group in static_feature_types.keys():
        counter = 0
        for feature in static_feature_types[feature_group].keys():
            immut = False
            if "race" in feature.lower() or "sex" in feature.lower():
                immut = True
            if static_feature_types[feature_group][feature][1] > 2 and static_feature_types[feature_group][feature][0] == 'cat':
                for i in range(static_feature_types[feature_group][feature][1]):
                    if immut:
                        immutable.append(str((feature_group,counter)))
                    categorical.append(str((feature_group,counter)))
                    counter += 1
            elif static_feature_types[feature_group][feature][0] == 'con':
                if immut:
                    immutable.append(str((feature_group,counter)))
                continuous.append(str((feature_group,counter)))
                counter += 1
            else:
                if immut:
                    immutable.append(str((feature_group,counter)))
                categorical.append(str((feature_group,counter)))
                counter += 1
    for feature_group in assessments:         
        temp_cols = list(X_df[feature_group].columns)
        asses_cols = [feature_group]*len(temp_cols)
        continuous.extend(list(map(str, list(zip(asses_cols, temp_cols)))))
        
    return continuous, categorical, immutable