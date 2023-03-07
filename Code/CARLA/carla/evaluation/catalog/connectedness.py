import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from CARLA.carla.models.negative_instances import predict_negative_instances

from CARLA.carla.evaluation import remove_nans
from CARLA.carla.evaluation.api import Evaluation


class Connectedness(Evaluation):
    """
    Computes connectedness for generated counterfactuals

    Notes
    -----
    - Hyperparams
        * "min_samples": int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        * "eps": int
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        * "cf_label": int
            What class to use as a target.
    """

    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.min_samples = self.hyperparameters["min_samples"]
        self.eps = self.hyperparameters["eps"]
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["connectedness"]

    def _connectedness(self, counterfactuals):
        if self.mlmodel._fit_full_data:
            full_dataset = self.mlmodel.X.copy()
        else:
            full_dataset = self.mlmodel.X_train.copy()
        positive_indices = list(set(full_dataset.index) - set(predict_negative_instances(self.mlmodel, full_dataset).index))
        positive_factuals = full_dataset.loc[positive_indices].copy()
        positive_factuals = self.mlmodel.get_ordered_features(positive_factuals)
        dbscan_results = []

        for i in range(len(counterfactuals)):
            data = np.r_[positive_factuals.values, counterfactuals.values[i, :].reshape(1, -1)]
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
            dbscan_results.append(clustering.labels_[-1])
        
        return [1 if x!=-1 else x for x in dbscan_results]

    def get_evaluation(self, factuals, counterfactuals):
        counterfactuals_without_nans = remove_nans(counterfactuals)

        if counterfactuals_without_nans.empty:
            conns = np.nan
        else:
            conns = self._connectedness(counterfactuals=counterfactuals_without_nans)

        return pd.DataFrame(conns, columns=self.columns)
