"""This module contains all the functions executing steps of preprocessing, implying data manipulation (not observation). That's mostly used into V) part
"""




import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables



# --------------------------------------------------------------
#  Formatting X, Y and Splitting it into train, valid, test sets (used in 'V)1)')
# --------------------------------------------------------------
def formatting_splitting(dataset_restricted_0, col_to_delete_list, train_proportion, test_proportion, random_state_0, dataset_0):
    """  
    This function selects the data from dataset_restricted_0, removes rows where the nb of matchs played by teams is inferior to min_played_matchs_nb, removes column(s) of feature(s) we don't want to keep in our dataset (if there are), returns separated features and labels.
    Then it splits our data into train, valid and test sets and convert them into dataframes. It returns X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ready to be given tou our model and the calibrator.
    
    Args:  
        dataset_restricted_0 (DataFrame): Restricted dataset obtained in III)3)
        
        col_to_delete_list (list): (If there are) List of columns names we want to delete from dataset because we don't want the model to use it.
        
        train_proportion (int)
        
        test_proportion (int)
        
        dataset_0 (DataFrame): The dataset containing the data.
    
    Returns:
        Tuple: (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    """
    
    #Definition of Y
    #On  selectionne uniquement les lignes où le nb de match joués > min_played_matchs_nb (défini dans 'Definition of restricted datasets...')
    
    col1 = dataset_0[dataset_0['HT_played_matchs_nb']>constant_variables.min_played_matchs_nb]['RH']
    col2 = dataset_0[dataset_0['AT_played_matchs_nb']>constant_variables.min_played_matchs_nb]['RA']
    Df_concatenated_target_column = pd.concat([col1, col2], axis=0, ignore_index=True)
    
    #On convertit les deux colonnes concaténées en Dataframe
    Df_concatenated_target_column = pd.DataFrame(Df_concatenated_target_column, columns = ["Result"])

    (X,Y) = dataset_restricted_0, Df_concatenated_target_column
    
    #Supprimer les colonnes demandées
    for col in col_to_delete_list:
        X=X.drop(col, axis=1)

    
    # Convertir X et Y en tableaux NumPy pour les conformer au type de données pris en entrée par la fonction train_test_split
    X_np_values = X.values
    Y_np_values = Y.values
    
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_np_values,
                                                        Y_np_values,
                                                        test_size=(1-train_proportion),
                                                        random_state = random_state_0,
                                                        shuffle = True,
                                                        stratify = Y_np_values)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp,
                                                        Y_temp,
                                                        test_size=(test_proportion/(1-train_proportion)),
                                                        random_state = random_state_0,
                                                        shuffle = True,
                                                        stratify = Y_temp)



    #Convertir X_train, X_test, Y_train, Y_test, X_valid, Y_valid en Dataframes
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    Y_train = pd.DataFrame(Y_train, columns=Y.columns)
    Y_test = pd.DataFrame(Y_test, columns=Y.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    Y_valid = pd.DataFrame(Y_valid, columns=Y.columns)
    
    return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)


    
# --------------------------------------------------------------
#  Removing outliers (not used but kept in case... placed in 'V)3)')
# --------------------------------------------------------------

#Info from features boxplot analysis that aims at identifying outliers:

#HT_avg_victory_pm: seems OK
#HT_avg_pm_points_ponderated_by_adversary_perf: OK
#HT_avg_goal_diff_pm: OK
#HT_avg_scored_g_conceedded_g_ratio: OK
#HT_avg_collected_points_pm: OK
#HT_ranking: OK
#HT_annual_budget: OK
#Points_HT_5lm_PM: OK
#GoalDiff_HT_5lm_PM: OK
#HT_5lm_week_ranking: OK
#HT_Diff_avg_corners: OK
#HT_avg_shots_nb: OK
#HT_avg_shots_on_target_nb: OK
#HT_avg_fouls_nb: OK
#HT_avg_possession: OK but: when i plot the mean on the boxplot it is equal to 50%, so normal. But when i do it manually i get 48%...
#HT_avg_xg
#HT_avg_odds_victory_proba: OK


#Delete the outliers of every features in dataset.
def outliers_removal(X_0, Y_0, iqr_multiplier):
    """  
    This function removes outliers from the provided DataFrames for all features (identifying outliers for each column).
    This function is not currently used, as I believe that all our data is informative for our model. In our context, outliers are not inaccurate data; all values represent realistic scenarios. This step was made to test if it could improve our model perf. I have tested, I don't remember of the results but i guess it was not soo incredible as i chosed not to used it.
    
    Args:
        X_0 (DataFrame): DataFrame containing the features for which outliers should be removed.
        
        Y_0 (DataFrame): Dataframe of the labels/targets corresponding to the X_0 Dataframe
        
        iqr_multiplier (float):  A multiplier used to determine the outlier boundaries. Outlier bounds are calculated as follows:lower_bound =   q1 - (iqr_multiplier * iqr)    upper_bound = q3 + (iqr_multiplier * iqr)
    
    Returns:
        Tuple: (clean_data, clean_target) X_0 and Y_0 without the rows where we identified outliers.
    """
    clean_data = X_0
    clean_target = Y_0
    rows_to_remove = []
    X_col_names = X_0.columns
    for i in range(X_0.shape[1]):
        #identifying outliers
        q1, q3 = np.percentile(X_0[X_col_names[i]], [25,75])
        iqr = q3-q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        # Find the rows that contain outliers
        outliers_rows = (clean_data[X_col_names[i]] < lower_bound) | (clean_data[X_col_names[i]] > upper_bound)
        rows_to_remove.extend(clean_data[outliers_rows].index)
    # Remove duplicates and sort indices to delete
    rows_to_remove = sorted(list(set(rows_to_remove)))
    # Drop outliers from clean_data
    clean_data = clean_data.drop(rows_to_remove)
    # Drop corresponding rows from clean_target
    clean_target = clean_target.drop(rows_to_remove)

    return(clean_data, clean_target)
