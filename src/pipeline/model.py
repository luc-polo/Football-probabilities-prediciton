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
features_selected_01 = ["Diff_Avg_victory",
                     "Diff_Avg_points_pm_ponderated_by_adversary_perf",
                     "Diff_Avg_goal_diff", 
                     "Diff_Avg_scored_g_conceeded_g_ratio",
                     "Diff_Avg_collected_points", 
                     "Diff_Week_ranking",
                     "Diff_Annual_budget",
                     "Diff_Points_5lm",
                     "Diff_Goal_Diff_5lm",
                     'Diff_Week_ranking_5lm',
                     'Diff_avg_corners_nb',
                     'Diff_Avg_shots_nb',
                     'Diff_Avg_shots_on_target_nb',
                     'Diff_Avg_fouls_nb',
                     'Diff_Avg_possession',
                     'Diff_Avg_odds_victory_proba',
                     'H_A_status']

pipeline_01 = Pipeline(steps=[ ("scaler", scaler),  ("model", model_01)])

# --------------------------------------------------------------
# Pipeline 2 (perf = 4.064 before week_ranking changes)
# -------------------------------------------------------------- 
model_02 = LogisticRegression(penalty = 'l2', C = 0.0177, fit_intercept=True, random_state = 999, solver = 'saga', max_iter= 3000, verbose = False, tol=1e-4)
features_selected_02 = ["Diff_Avg_points_pm_ponderated_by_adversary_perf",
                        "Diff_Avg_goal_diff", 
                        "Diff_Avg_scored_g_conceeded_g_ratio",
                        "Diff_Avg_collected_points",
                        "Diff_Annual_budget",
                        "Diff_Points_5lm",
                        "Diff_Goal_Diff_5lm",
                        'Diff_Week_ranking_5lm',
                        'Diff_avg_corners_nb',
                        'Diff_Avg_shots_nb',
                        'Diff_Avg_shots_on_target_nb',
                        'Diff_Avg_fouls_nb',
                        'Diff_Avg_possession',
                        'Diff_Avg_xg',
                        'H_A_status']

pipeline_02 = Pipeline(steps=[ ("scaler", scaler),  ("model", model_02)])

# --------------------------------------------------------------
# Pipeline 3,  (perf = 4.108 after week_ranking changes)
# -------------------------------------------------------------- 

features_selected_03 = ['Diff_Avg_points_pm_ponderated_by_adversary_perf',
                        'Diff_Avg_goal_diff',
                        'Diff_Avg_scored_g_conceeded_g_ratio',
                        'Diff_Avg_collected_points',
                        'Diff_Annual_budget',
                        'Diff_Points_5lm',
                        'Diff_Goal_Diff_5lm',
                        'Diff_avg_corners_nb',
                        'Diff_Avg_shots_nb',
                        'Diff_Avg_shots_on_target_nb',
                        'Diff_Avg_fouls_nb',
                        'Diff_Avg_possession',
                        'Diff_Avg_xg',
                        'H_A_status']
pipeline_03 = Pipeline(steps=[('scaler', StandardScaler()),
                ('features_selector', SelectKBest(k=14)),
                ('model',
                 LogisticRegression(C=0.025118864315095808, max_iter=3000,
                                    random_state=999, solver='saga',
                                    verbose=False))])



['Avg_victory', 'Avg_points_pm_ponderated_by_adversary_perf',
       'Avg_goal_diff', 'Avg_scored_g_conceeded_g_ratio',
       'Avg_collected_points', 'Week_ranking', 'Annual_budget', 'Points_5lm',
       'Goal_Diff_5lm', 'Week_ranking_5lm', 'Diff_avg_corners_nb',
       'Avg_shots_nb', 'Avg_shots_on_target_nb', 'Avg_possession',
       'Avg_odds_victory_proba', 'H_A_status']
