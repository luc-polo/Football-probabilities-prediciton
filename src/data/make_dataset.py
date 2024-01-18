"""
This module contains the function that creates the raw data dataframe.
"""

import pandas as pd
from dateutil.parser import parse
import sys
import os

#modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data import constant_variables
import useful_functions


# --------------------------------------------------------------
# Define CSV files paths
# --------------------------------------------------------------
adr1="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2014-to-2015-stats.csv"
adr2="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2015-to-2016-stats.csv"
adr4="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2017-to-2018-stats.csv"
adr3="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2016-to-2017-stats.csv"
adr5="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2018-to-2019-stats.csv"
adr6="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2019-to-2020-stats.csv"
adr7="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2020-to-2021-stats.csv"
adr8="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2021-to-2022-stats.csv"
adr9="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/france-ligue-1-matches-2022-to-2023-stats.csv"


# --------------------------------------------------------------
# Convert CSV into DataFrame
# --------------------------------------------------------------
def read_data(files):
    """
        Taking in entry the paths of the csv files, it outputs a DataFrame that is the concatenation of all our data.
        During this process the function converts 'date_GMT' column do datetime format, delete the raws of matchs definitivly.

    Args:
        files (list): The list of paths towards csv files that contain the data of one season for one championship. That's a list of str.
        
    Returns:
        DataFrame: The DataFrame that contains the raw statistics of each match for the season(s) of the championship(s) we inputed the path in 'files' list
    
    Exemples:
        >>> read_data([adr1, adr2, adr3, adr4])
        DataFrame
    """
    #Read CSV files and concat them
    dataset= pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    # Working with datetimes
    dataset["date_GMT"]=dataset["date_GMT"].apply(lambda x: parse(x))
    
    #Suppresession des lignes où le match a été annulé (covid)
    rows_to_keep = dataset["status"] != "canceled"
    dataset = dataset[rows_to_keep].reset_index(drop=True)
    
    
    return(dataset)

#dataset = read_data(files)

# --------------------------------------------------------------
# Define a function to load the dataframe saved at data/interim/data_interim.pkl
# --------------------------------------------------------------

def load_data(seasons_present_in_df_info):
    """ 
    This function imports/loads the DataFrame we will name dataset, from data/interim/data_interim.
    It says what are the seasons represented in this dataframe if asked.
    """
    
    daframe_location_path = 'C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/interim/data_interim.pkl'
    # Importation of dataset
    dataset = pd.read_pickle(daframe_location_path)
    
    if seasons_present_in_df_info == True:
        #Make out what are the seasons represented in this dataframe
        seasons_in_dataframe = dataset['date_GMT'].dt.year.unique()
        # Remove the holdest year which is only the beginning of the first season
        min_value = min(seasons_in_dataframe)
        seasons_in_dataframe = seasons_in_dataframe[seasons_in_dataframe != min_value]
        print("This dataframe contains matchs of the seasons: ", seasons_in_dataframe)

    return dataset

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

def save_dataframe_into_data_interim(dataset_0):
    """ 
        This function saves the dataframe created with read_data() in data/interim. We just have to input the dataframe to save. The function deletes the old file at this location. It also displays wether the old and new dataframe saved into data/interim are the same.
    
    Args:
        dataset_0 (Dataframe): the dataframe we want to save
        
    Returns:
        None
    """

    # Define the absolute path of the dataset destination, which is also the path of the actual doc located there that we will delete
    dataset_destination_path = "C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/interim/data_interim.pkl"
    
    #We compare if the old data_interim.pkl file that we are gonna delete and dataset_0¨are exactly the same
    old_dataframe = load_data(seasons_present_in_df_info = False)
    if old_dataframe.equals(dataset_0):
        print("The old data_interim.pkl file, and the new one ARE the same")
    else:
        print("The old data_interim.pkl file, and the new one ARE NOT the same")
    
    
    # We delete the old file
    try: 
        os.remove(dataset_destination_path)
        print("The old 'data_interim.pkl' file was well deleted")
    except Exception as e:
        print(f"An error occurred while deleting the old datafram file: {e}")
        print("The old 'data_interim.pkl' file was not removed (probably because it does not exist in this location).")
        

    # Export the dataset to the path constructed
    try: 
        dataset_0.to_pickle(dataset_destination_path)
        print("The new 'data_interim.pkl' file was well saved \n")
    except Exception as e:
        print(f"An error occurred while saving the new dataframe: {e}")
        print("The 'data_interim.pkl' was not saved. \n")


# --------------------------------------------------------------
# Definition of restricted datasets with a selection of revelevant features
# --------------------------------------------------------------

def restricted_datasets(dataset_0):
    """ 
        This function returns restricted dataframes that contain subsets of features considered as relevant. We also gather HT and AT features in one column. We will use these DataFrames to explore their data and to test our model with. The function also eliminate the matchs where Game Week < min_played_matchs_nb (defined in constant_variables).
        The function still need to be completed. We must make a shorter subset of features.
    
    Args:
        dataset_0 (Dataframe): The dataframe we want to extract the columns from
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - names_col_concat_rest_ds_2: DataFrame with relevant subsets of 18 features.
            - names_col_concat_rest_ds_3: DataFrame with relevant subsets of 23 features.
        
    """
    #On définit un nb de matchs min joués pour selectionner les lignes du dataset qui seront ajoutées a ces nouveau df
    min_played_matchs_nb_0 = constant_variables.min_played_matchs_nb


    #Restricted dataset 1


    #Restricted dataset 3
    names_col_rest_ds_3 =  ["HT_avg_victory_pm", "AT_avg_victory_pm", 
                            "HT_avg_pm_points_ponderated_by_adversary_perf", "AT_avg_pm_points_ponderated_by_adversary_perf",
                            "HT_avg_goal_diff_pm","AT_avg_goal_diff_pm",
                            "HT_avg_scored_g_conceedded_g_ratio","AT_avg_scored_g_conceedded_g_ratio",
                            "HT_avg_collected_points_pm","AT_avg_collected_points_pm",
                            "HT_week_ranking","AT_week_ranking",
                            "annual budget of HT","annual budget of AT",
                            "Points_HT_5lm_PM","Points_AT_5lm_PM",
                            "GoalDiff_HT_5lm_PM","GoalDiff_AT_5lm_PM",
                            "HT_5lm_week_ranking","AT_5lm_week_ranking",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_avg_shots_nb","AT_avg_shots_nb",
                            "HT_avg_shots_on_target_nb","AT_avg_shots_on_target_nb",
                            "HT_avg_fouls_nb","AT_avg_fouls_nb",
                            "HT_avg_possession","AT_avg_possession",
                            "HT_avg_xg","AT_avg_xg",
                            "HT_avg_odds_victory_proba","AT_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status",
                            "HT_Diff_Points_3lm", "AT_Diff_Points_3lm",
                            "HT_Diff_Goal_Diff_3lm", "AT_Diff_Goal_Diff_3lm",
                            "HT_Diff_Points_1lm", "AT_Diff_Points_1lm",
                            "HT_Diff_Goal_Diff_1lm", "AT_Diff_Goal_Diff_1lm",
                            "HT_Diff_avg_fouls_nb", "AT_Diff_avg_fouls_nb"]
    names_col_concat_rest_ds_3=["Avg_victory",
                                "Avg_points_pm_ponderated_by_adversary_perf",
                                "Avg_goal_diff", 
                                'Avg_scored_g_conceeded_g_ratio',
                                'Avg_collected_points', 
                                'Week_ranking',
                                "Annual_budget", 
                                "Points_5lm", 
                                "Goal_Diff_5lm", 
                                'Week_ranking_5lm', 
                                'Diff_avg_corners_nb',
                                'Avg_shots_nb', 
                                'Avg_shots_on_target_nb', 
                                'Avg_fouls_nb', 
                                'Avg_possession', 
                                'Avg_xg', 
                                'Avg_odds_victory_proba', 
                                'H_A_status', 
                                "Diff_Points_3lm", 
                                "Diff_Goal_Diff_3lm", 
                                "Diff_Points_1lm", 
                                "Diff_Goal_Diff_1lm",
                                "Diff_avg_fouls_nb"]
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_3 = useful_functions.HT_AT_col_merger(names_col_rest_ds_3, names_col_concat_rest_ds_3, min_played_matchs_nb_0, dataset_0)  
                
                          
    #Restricted dataset 2
    names_col_rest_ds_2 = ["HT_avg_victory_pm", "AT_avg_victory_pm", 
                            "HT_avg_pm_points_ponderated_by_adversary_perf", "AT_avg_pm_points_ponderated_by_adversary_perf",
                            "HT_avg_goal_diff_pm","AT_avg_goal_diff_pm",
                            "HT_avg_scored_g_conceedded_g_ratio","AT_avg_scored_g_conceedded_g_ratio",
                            "HT_avg_collected_points_pm","AT_avg_collected_points_pm",
                            "HT_week_ranking","AT_week_ranking",
                            "annual budget of HT","annual budget of AT",
                            "Points_HT_5lm_PM","Points_AT_5lm_PM",
                            "GoalDiff_HT_5lm_PM","GoalDiff_AT_5lm_PM",
                            "HT_5lm_week_ranking","AT_5lm_week_ranking",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_avg_shots_nb","AT_avg_shots_nb",
                            "HT_avg_shots_on_target_nb","AT_avg_shots_on_target_nb",
                            "HT_avg_fouls_nb","AT_avg_fouls_nb",
                            "HT_avg_possession","AT_avg_possession",
                            "HT_avg_xg","AT_avg_xg",
                            "HT_avg_odds_victory_proba","AT_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status"
                            ]
    names_col_concat_rest_ds_2 =["Avg_victory","Avg_points_pm_ponderated_by_adversary_perf","Avg_goal_diff", 
                                'Avg_scored_g_conceeded_g_ratio','Avg_collected_points', 'Week_ranking',
                                "Annual_budget", "Points_5lm", "Goal_Diff_5lm", 'Week_ranking_5lm', 'Diff_avg_corners_nb',
                                'Avg_shots_nb', 'Avg_shots_on_target_nb', 'Avg_fouls_nb', 'Avg_possession', 'Avg_xg', 'Avg_odds_victory_proba',
                                'H_A_status']
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_2 = useful_functions.HT_AT_col_merger(names_col_rest_ds_2, names_col_concat_rest_ds_2, min_played_matchs_nb_0, dataset_0)
        
    return concat_restricted_ds_2, concat_restricted_ds_3