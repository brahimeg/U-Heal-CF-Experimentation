import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from CARLA.carla.evaluation import remove_nans
from CARLA.carla.evaluation.api import Evaluation


class SingleYNN(Evaluation):
    """
    Computes y-Nearest-Neighbours for generated counterfactuals

    Notes
    -----
    - Hyperparams

        * "y": int
            Number of neighbours to use.
        * "cf_label": int
            What class to use as a target.
    """

    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.y = self.hyperparameters["y"]
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["single-y-Nearest-Neighbours"]

    def _single_ynn(self, counterfactuals):

        factuals = self.mlmodel.get_ordered_features(self.mlmodel.data.df)

        ynns = []
        nbrs = NearestNeighbors(n_neighbors=self.y).fit(factuals.values)
        for i, row in counterfactuals.iterrows():
            if np.any(row.isna()):
                raise ValueError(f"row {i} did not contain a valid counterfactual")

            knn = nbrs.kneighbors(
                row.values.reshape((1, -1)), self.y, return_distance=False
            )[0]
            number_of_diff_labels = 0
            for idx in knn:
                neighbour = factuals.iloc[idx]
                neighbour = neighbour.values.reshape((1, -1))
                neighbour_label = np.argmax(self.mlmodel.predict_proba(neighbour))
                number_of_diff_labels += np.abs(self.cf_label - neighbour_label)
            score = 1 - (1 / (self.y)) * number_of_diff_labels
            ynns.append(score)

        return ynns

    def get_evaluation(self, factuals, counterfactuals):
        counterfactuals_without_nans = remove_nans(counterfactuals)

        if counterfactuals_without_nans.empty:
            ynns = np.nan
        else:
            ynns = self._single_ynn(counterfactuals=counterfactuals_without_nans)

        return pd.DataFrame(ynns, columns=self.columns)
