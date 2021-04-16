# Detecting-Parkinson-s-Disease-with-XGBoost
Detecting Parkinson’s Disease with XGBoost - Basic Model


1. sklearn.preprocessing import MinMaxScaler
    class sklearn.preprocessing.MinMaxScaler(feature_range=0, 1, *, copy=True, clip=False)[source]
Transform features by scaling each feature to a given range.
This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

The transformation is given by:
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
where min, max = feature_range.
This transformation is often used as an alternative to zero mean, unit variance scaling.


2. xgboost import XGBClassifier
    https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    
3. accuracy_score
    In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
