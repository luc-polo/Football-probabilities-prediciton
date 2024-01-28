"""This module contains all the function necessary to combine/merge the data of Footy Stats and Football Data"""


# --------------------------------------------------------------
# 
# --------------------------------------------------------------
def team_names_uniformisation(Football_data_dataset_0, Footy_Stats_dataset_0):
    """Uniformise the names of teams in Footy Stats and Football Data datasets
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


def find_corresponding_match():
    """Find the corresponding match in Football Data dataset
    """

def replace_col_values():
    """Replace values in the HT and AT columns of Footy Stats dataset by the values in Football Data values
    """

