from Carla.models.api import MLModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier


# Custom black-box models need to inherit from
# the MLModel interface

class CustomClf(MLModel):
    def __init__(self, data, clf, scaling=False, fit_full_data=False, calibration=None, bagging=None):
        super().__init__(data)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._backend = 'pytorch'
        self._fit_full_data = fit_full_data
        self._bagging = bagging
        self._calibration = calibration

        self._X = self.data.df[list(set(self.data.df_train.columns) - {self.data.target})].copy()
        self._y = self.data.df[self.data.target].copy()
        self._X_train = self.data.df_train[list(set(self.data.df_train.columns) - {self.data.target})].copy()
        self._X_test = self.data.df_test[list(set(self.data.df_test.columns) - {self.data.target})].copy()
        self._y_train = self.data.df_train[self.data.target].copy()
        self._y_test = self.data.df_test[self.data.target].copy()
        
        # you can not only use the feature input order to
        # order the data but also to e.g. restrict the input
        # to only the continous features
        self._feature_input_order = list(self._X_train.columns)
        self._feature_input_order.sort()
        
        # order data (column-wise) before training
        self._X =  self._X[self._feature_input_order]
        self._X_train = self._X_train[self._feature_input_order]
        self._X_test = self._X_test[self._feature_input_order]
            
        if bagging != None:
            clf = BaggingClassifier(clf, n_estimators=bagging, bootstrap=False, n_jobs=-1)
            
        if scaling:
            self._mymodel = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
        else:
            self._mymodel = Pipeline([('estimator', clf)])
            
        if self._fit_full_data:
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
            
        if calibration != None:
            if not scaling:
                self._mymodel = Pipeline([('estimator', CalibratedClassifierCV(self._mymodel['estimator'], 
                                                                            method=calibration, n_jobs=-1, cv="prefit"))])
            else:
                self._mymodel = Pipeline([('transformer', StandardScaler()), ('estimator', CalibratedClassifierCV(self._mymodel['estimator'], 
                                                                                method=calibration, n_jobs=-1, cv="prefit"))])
            
            if self._fit_full_data:
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
        return self._backend 
    
    @backend.setter
    def backend(self, x):
        self._backend = x

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel
    @raw_model.setter
    def raw_model(self, model):
        self._mymodel = model
    
    @property
    def model_type(self):
        return "ann"
    
    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        try:
            if type(x) == torch.Tensor:
                return torch.Tensor(self._mymodel.predict_proba(x.cpu())).to(self._device)
            else:
                return self._mymodel.predict_proba(x)
        except:
            if type(x) == torch.Tensor:
                return torch.tensor(self._mymodel.predict_proba(x.cpu().detach().numpy())).float().to(self._device)
            else:
                return self._mymodel.predict_proba(x.cpu())
