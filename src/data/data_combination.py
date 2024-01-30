"""This module contains all the function necessary to combine/merge the data of Footy Stats and Football Data"""

import pandas as pd

# --------------------------------------------------------------
# 
# --------------------------------------------------------------
def team_names_uniformisation(Football_data_dataset_0, Footy_Stats_dataset_0):
    """
        Uniformise the names of teams in Footy Stats and Football Data datasets
    
    Args:
        Football_data_dataset_0 (DataFrame): Dataframe containing data from Football Data website
        
        Footy_Stats_dataset_0 (DataFrame): Dataframe containing data from Footy Data website
    """
    # Change Football Data team names
    football_datat_team_names_to_change = {'Ajaccio GFCO' : 'Gazélec Ajaccio',
                                           'Paris SG': 'PSG',
                                           'Evian Thonon Gaillard': 'Evian'}
    
    for team_name in football_datat_team_names_to_change.keys():
        Football_data_dataset_0["HomeTeam"].replace(team_name, football_datat_team_names_to_change[team_name], inplace=True)
        Football_data_dataset_0["AwayTeam"].replace(team_name, football_datat_team_names_to_change[team_name], inplace=True)
    
    # Change Footy Stats team names
    footy_stats_team_names_to_change = {'Amiens SC' : 'Amiens',
                                        'Angers SCO': 'Angers',
                                        'Saint-Étienne': 'St Etienne',
                                        'Olympique Marseille': 'Marseille',
                                        'Olympique Lyonnais': 'Lyon',
                                        'Thonon Evian FC': 'Evian'}
    
    for team_name in footy_stats_team_names_to_change.keys():
        Footy_Stats_dataset_0["home_team_name"].replace(team_name, footy_stats_team_names_to_change[team_name], inplace=True)
        Footy_Stats_dataset_0["away_team_name"].replace(team_name, footy_stats_team_names_to_change[team_name], inplace=True)
    
    return Football_data_dataset_0, Footy_Stats_dataset_0


def find_corresponding_match(row_nb_in_footy_dataset, footy_dataset_0, football_data_dataset_0):
    """For a given match (referred with its row nb) in footy_dataset, find the corresponding match in Football Data dataset and return its index. This function is used in replace_col_values() to change some footy_dataset columns values with footabll_data_dataset values.
    
    Args:
        row_nb_in_footy_dataset (int): The row index of the match in footy_dataset_0 we want to find in football_data_dataset_0.
        
        footy_dataset_0 (DataFrame): The dataframe with 'Footy Stats' data.
        
        football_data_dataset_0 (DataFrame): The dataframe with 'Football Data' data.
    
    Returns:
        int : The row nb in football_data_dataset of the match we look for
    """
    date_of_the_match = footy_dataset_0.at[row_nb_in_footy_dataset,'date_GMT'].date()  # the date_GMT column a format dd/mm/yyyy, used below 
    
    dat_col_of_footaball_dataset = football_data_dataset_0['Date'].dt.date  # the Date column at format dd/mm/yyyy, used below
    
    # We identify all rows in football_data_dataset_0 that have the same date as the match of footy dataset we are looking for
    possible_rows_identified_with_dates = football_data_dataset_0[dat_col_of_footaball_dataset == date_of_the_match]
    
    row_nb_in_football_data_dataset = 0
    
    for index, row in possible_rows_identified_with_dates.iterrows():
        if row['HomeTeam'] == footy_dataset_0.at[row_nb_in_footy_dataset, 'home_team_name'] and row['AwayTeam'] == footy_dataset_0.at[row_nb_in_footy_dataset, 'away_team_name']:
            row_nb_in_football_data_dataset = index
            break   # Exit the loop once a match is found
    
    return row_nb_in_football_data_dataset
    

def replace_col_values(footy_col_name, football_data_col_name, footy_dataset_0, football_data_dataset_0):
    """Replace values in the HT and AT columns of Footy Stats dataset by the values in Football Data values
    """
    nb_of_values_modified = 0
    nb_of_values_not_modified = 0
    for index, value in footy_dataset_0[footy_col_name].items():
        # Find the corresponding match row in Football Data dataset
        row_index_in_football_data_dataset = find_corresponding_match(index, footy_dataset_0, football_data_dataset_0)
        
        # If value in Football Data dataset in Nan we do not change
        if pd.isna(football_data_dataset_0.at[row_index_in_football_data_dataset, football_data_col_name]):
            nb_of_values_not_modified+=1
            pass # Do nothing
        # else replace the value in footy dataset by the one in football data dateset
        else:
            footy_dataset_0.at[index, footy_col_name] = football_data_dataset_0.at[row_index_in_football_data_dataset, football_data_col_name]
            nb_of_values_modified +=1
            
    # Print a little report of the modification made
    print(f"In the column {footy_col_name} of footy_dataset, {nb_of_values_modified} values were modified, {nb_of_values_not_modified} were not because the corresponding value in football_data_dataset was Nan.")
    
    return footy_dataset_0
        
    
    

