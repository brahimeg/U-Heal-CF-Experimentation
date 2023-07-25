import timeit
from typing import List

import pandas as pd
import numpy as np

from CARLA.carla.evaluation.api import Evaluation
from CARLA.carla.models.api import MLModel
from CARLA.carla.recourse_methods.api import RecourseMethod
class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain.
    recourse_method: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark.
    factuals: pd.DataFrame
        Instances for which we want to find counterfactuals.
    """

    def __init__(
        self,
        mlmodel: MLModel,
        factuals: pd.DataFrame = None,
        recourse_method: RecourseMethod = None,
        counterfactuals: pd.DataFrame = pd.DataFrame(),
        single_mode : bool = False
    ) -> None:

        self.mlmodel = mlmodel

        start = timeit.default_timer()
        if counterfactuals.empty:
            self._factuals = self.mlmodel.get_ordered_features(factuals.copy())
            self._recourse_method = recourse_method
            self._counterfactuals = recourse_method.get_counterfactuals(self._factuals)
            if single_mode:
                self._factuals = self.mlmodel.get_ordered_features(pd.DataFrame(np.repeat(factuals.copy().values, len(self._counterfactuals), axis=0), columns=factuals.columns))
        else:
            self._factuals = None
            self._counterfactuals = counterfactuals
        stop = timeit.default_timer()
        self.timer = stop - start

    def run_benchmark(self, measures: List[Evaluation]) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Parameters
        ----------
        measures : List[Evaluation]
            List of Evaluation measures that will be computed.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            measure.get_evaluation(
                counterfactuals=self._counterfactuals, factuals=self._factuals
            )
            for measure in measures
        ]
        output = pd.concat(pipeline, axis=1)
        if self._counterfactuals.dropna().empty:
            output.index = [-1]
        else:
            output.index = self._counterfactuals.dropna().index
        return output


