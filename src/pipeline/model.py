"""This module contains the different module we will or have tested"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------
# Pipeline 1
# --------------------------------------------------------------
scaler = StandardScaler()

filter_feat_selector = SelectKBest(f_classif, k= 16)

model_01 = LogisticRegression(penalty = 'l2', C = 0.0486967525, fit_intercept=True, random_state = 999, solver = 'sag', class_weight = {1:0.6, 0:0.4}, max_iter= 3000, verbose = False, tol=1e-4)

pipeline_01 = Pipeline(steps=[ ("scaler", scaler), ("features_selector",filter_feat_selector), ("model", model_01)])