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
        self.ensemble = self.hyperparameters["ensemble"]
        self.columns = ["Stability"]
        self.mlmodel = mlmodel


    def get_evaluation(self, counterfactuals, factuals):
        counterfactuals_without_nans, _ = remove_nans(
            counterfactuals, factuals
        )

        if counterfactuals_without_nans.empty:
            stabilities = pd.Series()
        else:
            stabilities = self._stability(
                counterfactuals_without_nans
            )

        return pd.DataFrame(stabilities, columns=self.columns)
    
    def _stability(self, counterfactuals):
        if self.ensemble:
            try:
                estimators = self.mlmodel.raw_model['estimator'].estimators_
                preds = np.array([x.predict(counterfactuals) for x in estimators])
                stabilities = np.mean(preds, axis=0)
            except:
                raise('Stability ensemble only supports sklearn.ensemble.BaggingClassifier!')
        else:
            all_preds = pd.DataFrame(index=counterfactuals.index)
            all_preds['orginal_label'] = self.mlmodel.predict(counterfactuals)
            all_preds.index=counterfactuals.index
            for i in range(self.threshold):
                print(i)
                if "CalibratedClassifierCV" in str(self.mlmodel.raw_model):
                    new_model = CustomClf(self.mlmodel.data, self.mlmodel.raw_model['estimator'].base_estimator.base_estimator,
                                            fit_full_data = self.mlmodel._fit_full_data, calibration = self.mlmodel._calibration, bagging = self.mlmodel._bagging)
                else:   
                    new_model = CustomClf(self.mlmodel.data, self.mlmodel.raw_model['estimator'],
                                        self.mlmodel._fit_full_data)
                all_preds[f'pred_{i}'] = new_model.predict(counterfactuals)
            stabilities = [row.sum()/self.threshold for row in all_preds.iloc[:,1:].values]  
        return pd.Series(stabilities)
