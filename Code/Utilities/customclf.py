from CARLA.carla.models.api import MLModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch

# Custom black-box models need to inherit from
# the MLModel interface

class CustomClf(MLModel):
    def __init__(self, data, clf, fit_full_data=False):
        super().__init__(data)

        self._X = self.data.df[list(set(self.data.df_train.columns) - {self.data.target})]
        self._y = self.data.df[self.data.target]
        self._X_train = self.data.df_train[list(set(self.data.df_train.columns) - {self.data.target})]
        self._X_test = self.data.df_test[list(set(self.data.df_test.columns) - {self.data.target})]
        self._y_train = self.data.df_train[self.data.target]
        self._y_test = self.data.df_test[self.data.target]
        
        # you can not only use the feature input order to
        # order the data but also to e.g. restrict the input
        # to only the continous features
        self._feature_input_order = list(self._X_train.columns)
        self._feature_input_order.sort()
        
        # order data (column-wise) before training
        self._X =  self._X_train[self._feature_input_order]
        self._X_train = self._X_train[self._feature_input_order]
        self._X_test = self._X_test[self._feature_input_order]

        self._mymodel = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
        if fit_full_data:
            rand_idx = np.random.permutation(self._X.shape[0])
            self._X = self._X.iloc[rand_idx,:]
            self._y = self._y.iloc[rand_idx]
            self._mymodel.fit(
                    self._X,
                    self._y
                )            
        else:
            rand_idx = np.random.permutation(self._X_train.shape[0])
            self._X_train = self._X_train.iloc[rand_idx,:]
            self._y_train = self._y_train.iloc[rand_idx]
            self._mymodel.fit(
                    self._X_train,
                    self._y_train
                )
            
    @property
    def X(self):
         return self._X
     
    @property
    def y(self):
         return self._y
        
    @property
    def X_train(self):
         return self._X_train
     
    @property
    def y_train(self):
         return self._y_train

    @property
    def X_test(self):
         return self._X_test
     
    @property
    def y_test(self):
         return self._y_test

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
         return self._feature_input_order

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        try:
            return self._mymodel.predict_proba(x)
        except RuntimeError:
            return torch.tensor(self._mymodel.predict_proba(x.detach().numpy())).float()