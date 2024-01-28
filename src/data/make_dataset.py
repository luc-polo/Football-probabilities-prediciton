"""
This module contains the function that creates the raw data dataframe.
"""

import pandas as pd
from dateutil.parser import parse
from datetime import date, datetime
import sys
import os

#modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables
import useful_functions


# --------------------------------------------------------------
# Define CSV files paths
# --------------------------------------------------------------
footy_2015_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2014-to-2015-stats.csv"
footy_2016_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2015-to-2016-stats.csv"
footy_2017_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2016-to-2017-stats.csv"
footy_2018_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2017-to-2018-stats.csv"
footy_2019_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2018-to-2019-stats.csv"
footy_2020_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2019-to-2020-stats.csv"
footy_2021_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2020-to-2021-stats.csv"
footy_2022_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2021-to-2022-stats.csv"
footy_2023_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2022-to-2023-stats.csv"


football_data_2015_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2014-2015_ligue1.csv"
football_data_2016_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2015-2016_ligue1.csv"
football_data_2017_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2016-2017_ligue1.csv"
football_data_2018_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2017-2018_ligue1.csv"
football_data_2019_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2018-2019_ligue1.csv"
football_data_2020_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2019-2020_ligue1.csv"
football_data_2021_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2020-2021_ligue1.csv"
football_data_2022_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2021-2022_ligue1.csv"
football_data_2023_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2022-2023_ligue1.csv"


# --------------------------------------------------------------
# Convert CSV into DataFrame
# --------------------------------------------------------------
def read_data(files, Footy_or_Football):
    """
        Taking in entry the paths of the csv files, it outputs a DataFrame that is the concatenation of all our data.
        During this process the function converts 'date_GMT' column do datetime format, delete the raws of matchs definitivly.

    Args:
        files (list): The list of paths towards csv files that contain the data of one season for one championship. That's a list of str.
        
        Footy_or_Football (str): Wether the csv data unputed is from Footy Stats website or Football Data website
        
    Returns:
        DataFrame: The DataFrame that contains the raw statistics of each match for the season(s) of the championship(s) we inputed the path in 'files' list
    
    Exemples:
        >>> read_data([adr1, adr2, adr3, adr4])
        DataFrame
    """
    #Read CSV files and concat them
    dataset= pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    # Working with datetimes
    if Footy_or_Football == 'footy':
        dataset["date_GMT"]=dataset["date_GMT"].apply(lambda x: parse(x))
    else:
        dataset["Date"] = dataset["Date"].astype(str)
        print(dataset["Date"].isna().sum())
        dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

    
    if Footy_or_Football == 'footy':
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


