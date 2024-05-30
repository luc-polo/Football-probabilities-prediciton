"""This module contains all the functions executing steps of preprocessing, implying data manipulation (not observation). That's mostly used into V) part
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


# modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables
import useful_functions


# --------------------------------------------------------------
#  Formatting X, Y and Splitting it into train, valid, test sets (used in 'V)1)')
# --------------------------------------------------------------


#Funciton for formatting ant cleaning data
def formatting_cleaning( H_A_col_to_concat_0, names_col_concatenated_0, col_to_remove, contextual_col, dataset_0):
    """
    This function selects a restricted number of features and concatenates Home and Away features for them, applying a filter on minimum GW. It also creates a X_info dataset with all the contextual features we don't want to input in our model.

    Args:
        H_A_col_to_concat_0 (list): List of column names we want to include in the final dataset for our pipeline. It contains the Home and Away teams col names that we will concatenate.
        
        names_col_concatenated_0 (list): List of names we will assign to the concatenated col (H_A_col_to_concat).
        
        col_to_remove (list): List of column names to be deleted
        
        contextual_col (list): List with the names of the concatenated columns containing a contextual information. That's the col we do not want to give to our model.
        
        dataset_0 (DataFrame): The feat_engineered_ds to be cleaned and formatted
         

    Returns:
        Tuple: (X_0, X_0_info, Y_0) X_0 is the formatted/clean features dataset and Y_0 the one of the target. X_0_info contains the contextual features (teams names, dates...)
    """
    Y_0 = useful_functions.HT_AT_col_merger(['RH', 'RA'], ['Result'], constant_variables.min_played_matchs_nb, dataset_0)
    X_0 = useful_functions.HT_AT_col_merger(H_A_col_to_concat_0, names_col_concatenated_0, constant_variables.min_played_matchs_nb, dataset_0)
    
    
    #Supprimer les colonnes demandées
    for col in col_to_remove:
        X_0.drop(col, axis=1, inplace = True)
        
    #We stock and delete the col containing contextual info on matchs (teams names, dates...)
    X_0_info = X_0[contextual_col]

    for col in contextual_col:
        X_0.drop(col, axis=1, inplace=True)
        
    return X_0, X_0_info, Y_0

# Here was the old function 'formatting_splitting_shuffle()'


# Function that makes a splitting based on seasons
def splitting(X_0, X_0_info, Y_0, test_seasons, train_seasons):
    """
        This function splits the formatted datasets into train and test sets, based on seasons.

    Args:
        X_0 (DataFrame): The formatted features dataset.
        
        X_0_info (DataFrame): The contextual information dataset.
        
        test_seasons (list): List of seasons we want to include in the test set.
        
        train_seasons (list): List of seasons we want to include in the train set
        
        dataset_0 (DataFrame): The original dataset.

    Returns:
        Tuple: (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test) 
        X_ are the formatted/clean features datasets and Y_ the ones of the targets. 
        X_info contain the contextual features (teams names, dates...)
    """
    
    # Split the data based on seasons
    
    # Define a condition that selects the indices of matches where season_year is in test_seasons
    test_date_condition = (X_0["Season_year"].isin(test_seasons))
    
    X_test = X_0[test_date_condition]
    X_test_info = X_0_info[test_date_condition]
    Y_test = Y_0[test_date_condition]
    
    # Define a conditoin that selects the lines index of matches of the test seasons
    train_date_condition = (X_0["Season_year"].isin(train_seasons))
    
    X_train = X_0[train_date_condition]
    X_train_info = X_0_info[train_date_condition]
    Y_train = Y_0[train_date_condition]
    
     # Test 1: Check if the total number of rows in X_test and X_train equals the number of rows in X_0
    assert len(X_test) + len(X_train) == len(X_0), "The total number of rows in X_test and X_train does not equal the number of rows in X_0."

    # Test 2: Check if there are no common rows between X_test and X_train
    common_rows = pd.concat([X_test, X_train]).duplicated(keep=False)
    assert not common_rows.any(), "There are common rows between X_test and X_train."

    return (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test)


# Displaying the histogram of samples in train and test sets depending on the season
def hist_seasons(X_train_0, X_test_0):
    seasons = sorted(set(pd.concat([X_train_0['Season_year'], X_test_0['Season_year']], axis=0)))
    bin_edges = seasons + [max(seasons) + 1]
    plt.figure(figsize=(7,5))
    plt.hist([X_train_0['Season_year'], X_test_0['Season_year']], label = ['Train set', 'Test set'], bins = bin_edges, align='left', rwidth=0.5)
    plt.xlabel('Season')
    plt.ylabel('Number of samples')
    plt.title('Seasons Distribution between Train and Test Sets')
    plt.legend()
    plt.grid()
    plt.show()
    print(len(set(X_train_0['Season_year'])))
        
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

def imputation(feature_to_impute_name, features_list, X_0):
    """Apply imputation (KNNImputer) to a (whole) specific column. The imputation computation is performed using a list of features input into the function parameters.
    
    Args:
        feature_to_impute_name (str): The name of the feature we want to impute
        features_list (list): The list that contains the names of the features we want to input into KNNImputer in order to impute missing values for the feature_to_impute_name
        X_0 (DataFrame): dataset containing the feature we want to impute
        
    Returns:
        DataFrame: X_0 where we only imputed missing values of feature_to_impute_name
    """

    # Make 2 copy of the original dataset
    features_list.append(feature_to_impute_name)
    
    X_before_imputation = X_0.copy()
    X_to_impute = X_0.copy()[features_list]

    X_col_names = X_to_impute.columns
    
    # Apply the KNNImputer on X_to_impute
    imputer = KNNImputer(missing_values = 0, n_neighbors=5, weights='distance')
    fully_imputed_X = imputer.fit_transform(X_to_impute)
    fully_imputed_X = pd.DataFrame(fully_imputed_X, columns = X_col_names)

    # Replace the column in the original dataset with the feature for which we wanted to perform imputation.
    X_before_imputation[feature_to_impute_name] = fully_imputed_X[feature_to_impute_name]
    
    return(X_before_imputation)