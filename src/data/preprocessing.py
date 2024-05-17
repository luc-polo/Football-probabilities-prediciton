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

#Class that gether almost all parameters required in formatting_splitting... funcitons

class DataFormating_choices:
    def __init__(self, names_col_to_concat, names_col_concatenated, col_to_remove, contextual_col, test_seasons):
        self.names_col_to_concat = names_col_to_concat
        self.names_col_concatenated = names_col_concatenated
        self.col_to_remove = col_to_remove
        self.contextual_col = contextual_col
        self.test_seasons = test_seasons
....
A FINIR !!!!!



#Funciton for formatting ant cleaning data
def formatting_cleaning( H_A_col_to_concat, col_concatenated_names, col_to_delete_list, contextual_col, dataset_0):
    """This function is used in other functions as the two following. It contains all the what the formatting_splitting functions do, excepted the train_test_plit step.
    This function selects a restricted number of features and concatenates Home and Away features for them, applying a filter on minimum GW. It as well as creates a X_info dataset with all the contextual features we don't want to input in our model.

    Args:
        H_A_col_to_concat (list): List of column names we want to include in the final dataset for our pipeline. It contains the Home and Away teams col names that we will concatenate.
        
        col_concatenated_names (list): List of names we will assign to the concatenated col (H_A_col_to_concat).
        
        col_to_delete_list (list): List of column names to be deleted
        
        contextual_col (list): List with the names of the concatenated columns containing a contextual information. That's the col we do not want to give to our model.
        
        dataset_0 (DataFrame): The original dataset
         

    Returns:
        Tuple: (X_0, X_0_info, Y_0) X_0 is the formatted/clean features dataset and Y_0 the one of the target. X_0_info contains the contextual features (teams names, dates...)
    """
    Y_0 = useful_functions.HT_AT_col_merger(['RH', 'RA'], ['Result'], constant_variables.min_played_matchs_nb, dataset_0)
    X_0 = useful_functions.HT_AT_col_merger(H_A_col_to_concat, col_concatenated_names, constant_variables.min_played_matchs_nb, dataset_0)
    
    
    #Supprimer les colonnes demandées
    for col in col_to_delete_list:
        X_0.drop(col, axis=1, inplace = True)
        
    #We stock and delete the col containing contextual info on matchs (teams names, dates...)
    X_0_info = X_0[contextual_col]

    for col in contextual_col:
        X_0.drop(col, axis=1, inplace=True)
        
    return X_0, X_0_info, Y_0

# Function that shuffle data in the splitting process
def formatting_splitting_shuffle(H_A_col_to_concat, col_concatenated_names, col_to_delete_list, contextual_col, random_state_0, dataset_0, test_proportion, *train_proportion):
    """  
    This function selects the features H_A_col_to_concat, concatenates HT and AT col, removes rows where the nb of matchs played by teams is inferior to min_played_matchs_nb, removes column(s) of feature(s) we don't want to keep in our dataset (if there are), returns separated features and labels.
    Then it splits our data into train, (valid if needed) and test sets and convert them into dataframes. It returns X_train, Y_train, X_test, Y_test, (X_valid, Y_valid) ready to be given to our model and the calibrator. X_train_info, X_test_info, and (X_valid_info) contain contextual information such as team names, which we may need after executing the pipeline to analyze the coherence of the predicted probabilities.
    
    Args:
        H_A_col_to_concat (list): The list containing the names of HT and AT col we want to put into X_train and X_test
        
        col_concatenated_names (list): The list containing the names we we will give to the columns of the concatenation of H_A_col_to_concat
      
        dataset_restricted_0 (DataFrame): Restricted dataset obtained in III)3)
        
        col_to_delete_list (list): (If there are) List of columns names we want to delete from dataset because we don't want the model to use it.
        
        contextual_col (list): List with the names of the concatenated columns containing a contextual information. That's the col we do not want to give to our model.
        
        dataset_0 (DataFrame): The dataset containing the data.
        
        test_proportion (int): The prooportion of the dataset we want to dedicate to the test set
                
        *train_proportion (int): To precise only if we want to make a validation set. The prooportion of the dataset we want to dedicate to the train set
    
    Returns:
        Tuple: (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test, *X_valid_info, *X_valid, *Y_valid)
    """
    
    #Definition of Y and X
    #On  selectionne uniquement les lignes où le nb de match joués > min_played_matchs_nb (défini dans 'Definition of restricted datasets...') On concatenne les HT et AT col
    
    Df_concatenated_target_column = useful_functions.HT_AT_col_merger(['RH', 'RA'], ['Result'], constant_variables.min_played_matchs_nb, dataset_0)
    
    Df_concatenated_features_column = useful_functions.HT_AT_col_merger(H_A_col_to_concat, col_concatenated_names, constant_variables.min_played_matchs_nb, dataset_0)

    (X,Y) = Df_concatenated_features_column, Df_concatenated_target_column
    
    #Supprimer les colonnes demandées
    for col in col_to_delete_list:
        X=X.drop(col, axis=1)
    
    # Convertir X et Y en tableaux NumPy pour les conformer au type de données pris en entrée par la fonction train_test_split
    X_np_values = X.values
    Y_np_values = Y.values
    
    # Split the data into training, validation, and test sets
    if train_proportion:
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
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X_np_values,
                                                            Y_np_values,
                                                            test_size=test_proportion,
                                                            random_state = random_state_0,
                                                            shuffle = True,
                                                            stratify = Y_np_values)
        

    #Convertir X_train, X_test, Y_train, Y_test, X_valid, Y_valid en Dataframes
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    Y_train = pd.DataFrame(Y_train, columns=Y.columns)
    Y_test = pd.DataFrame(Y_test, columns=Y.columns)
    if train_proportion:
        X_valid = pd.DataFrame(X_valid, columns=X.columns)
        Y_valid = pd.DataFrame(Y_valid, columns=Y.columns)
    
    #We stock and delete the col containing contextual info on matchs (teams names, dates...)
    X_train_info = X_train[contextual_col]
    X_test_info = X_test[contextual_col]
    if train_proportion:
            X_valid_info = X_valid[contextual_col]
            
    for col in contextual_col:
        X_train.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)
        if train_proportion:
            X_valid.drop(col, axis=1, inplace=True)

    if train_proportion:
        return (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test, X_valid_info, X_valid, Y_valid)
    else:
        return (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test)

# Function that makes a splitting based on seasons
def formatting_splitting_seasonally(H_A_col_to_concat, col_concatenated_names, col_to_delete_list, contextual_col, dataset_0, test_seasons, train_seasons):
    """
        This function splits the dataset into train and test sets based on seasons and performs data formatting

    Args:
        H_A_col_to_concat (list): List of column names we want to include in the final dataset for our pipeline. It contains the Home and Away teams col names that we will concatenate.
        
        col_concatenated_names (list): List of names we will assign to the concatenated col (H_A_col_to_concat).
        
        col_to_delete_list (list): List of column names to be deleted.
        
        contextual_col (list): List with the names of the concatenated columns containing a contextual information. That's the col we do not want to give to our model.
        
        dataset_0 (DataFrame): The original dataset.
        
        test_seasons (list): List of seasons we want to put in test set
        
        train_seasons (list): List of seasons we want to put in train set

    Returns:
        Tuple: (X_train_info, X_train, Y_train, X_test_info, X_test, Y_test) X_ are the formatted/clean features datasets and Y_ the ones of the targets. X_info contain the contextual features (teams names, dates...)
    """
    
    #We split the data following seasons:
    df_test = dataset_0[dataset_0['Season_year'].isin(test_seasons)]
    df_train = dataset_0[dataset_0['Season_year'].isin(train_seasons)]
    
    #Definition of Y and X
    #On selectionne uniquement les lignes où le nb de match joués > min_played_matchs_nb (défini dans 'Definition of restricted datasets...'). Et on concatenne les HT et AT col
    
    X_test, X_test_info, Y_test = formatting_cleaning( H_A_col_to_concat, col_concatenated_names, col_to_delete_list, contextual_col, df_test)
    
    X_train, X_train_info, Y_train = formatting_cleaning( H_A_col_to_concat, col_concatenated_names, col_to_delete_list, contextual_col, df_train)

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