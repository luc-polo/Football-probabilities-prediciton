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

model_01 = LogisticRegression(penalty = 'l2', C = 0.0486967525, fit_intercept=True, random_state = 999, solver = 'sag', max_iter= 3000, verbose = False, tol=1e-4)

features_selected_01 =['Avg_victory', 'Avg_points_pm_ponderated_by_adversary_perf',
       'Avg_goal_diff', 'Avg_scored_g_conceeded_g_ratio',
       'Avg_collected_points', 'Week_ranking', 'Annual_budget', 'Points_5lm',
       'Goal_Diff_5lm', 'Week_ranking_5lm', 'Diff_avg_corners_nb',
       'Avg_shots_nb', 'Avg_shots_on_target_nb', 'Avg_possession',
       'Avg_odds_victory_proba', 'H_A_status']

pipeline_01 = Pipeline(steps=[ ("scaler", scaler),  ("model", model_01)])