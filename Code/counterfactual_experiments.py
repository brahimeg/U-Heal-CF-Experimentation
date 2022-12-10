import CARLA.carla.evaluation.catalog as evaluation_catalog
from CARLA.carla.data.api import Data
from CARLA.carla.data.catalog import OnlineCatalog
from CARLA.carla.evaluation import Benchmark
from CARLA.carla.models.api import MLModel
from CARLA.carla.models.catalog import MLModelCatalog
from CARLA.carla.models.negative_instances import predict_negative_instances
from CARLA.carla.recourse_methods import *
from CARLA.carla.recourse_methods.api import RecourseMethod
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

def make_benchmark(data_name="adult", model_name="ann"):
    # get data and mlmodel
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_name, backend="tensorflow")

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    # get recourse method
    hyperparams = {"num": 1, "desired_class": 1}
    recourse_method = Dice(model, hyperparams)

    # make benchmark object
    benchmark = Benchmark(model, recourse_method, test_factual)

    return benchmark


def run_benchmark():
    benchmark = make_benchmark()
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


# if __name__ == '__main___':
df_bench, metrics = run_benchmark()
metrics[0].y
