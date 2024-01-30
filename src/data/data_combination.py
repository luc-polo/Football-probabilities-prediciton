"""This module contains all the function necessary to combine/merge the data of Footy Stats and Football Data"""


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
    """Find the corresponding match in Football Data dataset and return its index
    """
    date_of_the_match = footy_dataset_0.at[row_nb_in_footy_dataset,'date_GMT'].date()
    
    dat_col_of_footaball_dataset = football_data_dataset_0['Date'].dt.date
    
    # We identify all rows in football_data_dataset_0 that have the same date as the match of footy dataset we are looking for
    possible_rows_identified_with_dates = football_data_dataset_0[dat_col_of_footaball_dataset == date_of_the_match]
    
    row_nb_in_football_data_dataset = 0
    
    for index, row in possible_rows_identified_with_dates.iterrows():
        if row['HomeTeam'] == footy_dataset_0.at[row_nb_in_footy_dataset, 'home_team_name'] and row['AwayTeam'] == footy_dataset_0.at[row_nb_in_footy_dataset, 'away_team_name']:
            row_nb_in_football_data_dataset = index
            break   # Exit the loop once a match is found
    
    return row_nb_in_football_data_dataset
    

def replace_col_values():
    """Replace values in the HT and AT columns of Footy Stats dataset by the values in Football Data values
    """

