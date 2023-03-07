import pandas as pd
import numpy as np
from gower import gower_matrix, gower_topn, smallest_indices

from CARLA.carla.models.api import MLModel
from CARLA.carla.recourse_methods.api import RecourseMethod
from CARLA.carla.models.negative_instances import predict_negative_instances

from CARLA.carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
    merge_default_parameters
)

class NaiveGower(RecourseMethod):
    """_summary_

    Args:
        RecourseMethod (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    _DEFAULT_HYPERPARAMS = {"retries": 0}

    def __init__(self, mlmodel: MLModel, hyperparams=None) -> None:

        supported_backends = ["sklearn", "pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )
        self._mutables = [
            feature
            for feature in self._mlmodel.feature_input_order
            if feature not in self._immutables
        ]
        self._continuous = self._mlmodel.data.continuous
        self._retries = self._params["retries"]
        self._categorical_enc = encode_feature_names(
            self._mlmodel.data.categorical, self._mlmodel.feature_input_order
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._mlmodel.get_ordered_features(factuals)
        if self._mlmodel._fit_full_data:
            full_dataset = self._mlmodel.X.copy()
        else:
            full_dataset = self._mlmodel.X_train.copy()
        full_immutable_dataset = full_dataset[self._immutables].copy()
        cat_features = [x in self._mlmodel.data.encoder.get_feature_names_out(self._mlmodel.data.categorical) for x in full_immutable_dataset.columns]
        matrix = gower_matrix(full_immutable_dataset, cat_features=cat_features)
        positive_indices = list(set(full_dataset.index) - set(predict_negative_instances(self._mlmodel, full_dataset).index))
        positive_factuals = full_dataset.loc[positive_indices].copy()
        positive_factuals = self._mlmodel.get_ordered_features(positive_factuals)
        list_cfs = []
        for index, value in factuals.iterrows():
            results = smallest_indices(np.nan_to_num(matrix[index][positive_indices], nan=1), self._retries+1)
            for i in range(self._retries):
                best_cf = positive_factuals.iloc[results['index'][i+1]].copy()
                best_cf.name = index
                best_cf[self._immutables] = value[self._immutables]
                if self._mlmodel.predict(best_cf.values.reshape(1,len(best_cf)))[0] == 1:
                    list_cfs.append(best_cf)
                    break
                elif i == self._retries - 1:
                    list_cfs.append(best_cf)
                    break
                else:
                    continue
        df_cfs = check_counterfactuals(self._mlmodel, list_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs

