from typing import List

import numpy as np
import pandas as pd

from Carla.evaluation import remove_nans
from Carla.evaluation.api import Evaluation


class Sparsity(Evaluation):
    """
    Computes sparsity for each counterfactual
    """

    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["Sparsity"]

    def _compute_sparsity(
        self, factual: np.ndarray, counterfactual: np.ndarray
    ) -> int:
        mutable_count = len(self.mlmodel.data.continuous)+len(self.mlmodel.data.categorical) - len(self.mlmodel.data._immutables)
        changed_features = 0
        for col_idx in range(len(counterfactual)):
            # if feature is changed
            if abs(factual[col_idx] - counterfactual[col_idx]) > 0.000001:
                changed_features += 1
        return changed_features/mutable_count

    def _sparsity(
        self,
        factuals: pd.DataFrame,
        counterfactuals: pd.DataFrame,
    ) -> List[List[int]]:
        """
        Computes sparsity measure for every counterfactual.

        Parameters
        ----------
        factuals:
            Encoded and normalized factual samples.
        counterfactuals:
            Encoded and normalized counterfactual samples.

        Returns
        -------
        List with sparsity values per counterfactual sample
        """
        df_enc_norm_fact = factuals.reset_index(drop=True)
        df_cfs = counterfactuals.reset_index(drop=True)

        df_cfs["sparsity"] = df_cfs.apply(
            lambda x: self._compute_sparsity(
                df_enc_norm_fact.iloc[x.name].values,
                x.values,
            ),
            axis=1,
        )
        return df_cfs["sparsity"].values.reshape((-1, 1)).tolist()

    def get_evaluation(self, counterfactuals, factuals):
        counterfactuals_without_nans, factual_without_nans = remove_nans(
            counterfactuals, factuals
        )

        if counterfactuals_without_nans.empty:
            redundancies = pd.Series()
        else:
            redundancies = self._sparsity(
                factual_without_nans,
                counterfactuals_without_nans,
            )

        return pd.DataFrame(redundancies, columns=self.columns)
