from typing import List
from Utilities.customclf import CustomClf
import numpy as np
import pandas as pd
from IPython.display import display

from CARLA.carla.evaluation import remove_nans
from CARLA.carla.evaluation.api import Evaluation


class Stability(Evaluation):
    """
    Computes stability for each counterfactual
    """

    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.cf_label = self.hyperparameters["cf_label"]
        self.threshold = self.hyperparameters["threshold"]
        self.columns = ["Stability"]
        self.mlmodel = mlmodel


    def get_evaluation(self, counterfactuals, factuals):
        counterfactuals_without_nans, factual_without_nans = remove_nans(
            counterfactuals, factuals
        )

        if counterfactuals_without_nans.empty:
            stabilities = np.nan
        else:
            stabilities = self._stability(
                factual_without_nans,
                counterfactuals_without_nans,
            )

        return pd.DataFrame(stabilities, columns=self.columns)
    
    def _stability(self, factuals, counterfactuals):
        all_preds = pd.DataFrame(index=counterfactuals.index)
        all_preds['orginal_label'] = self.mlmodel.predict(counterfactuals)
        all_preds.index=counterfactuals.index
        
        for i in range(self.threshold):
            new_model = CustomClf(self.mlmodel.data, self.mlmodel.raw_model['estimator'],
                                  self.mlmodel._fit_full_data)
            all_preds[f'pred_{i}'] = new_model.predict(counterfactuals)
        stabilities = [row.sum() == self.threshold+1 for row in all_preds.values]  
        return pd.Series(stabilities).astype(int)
