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
footy_2024_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Footy/france-ligue-1-matches-2023-to-2024-stats.csv"


football_data_2015_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2014-2015_ligue1.csv"
football_data_2016_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2015-2016_ligue1.csv"
football_data_2017_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2016-2017_ligue1.csv"
football_data_2018_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2017-2018_ligue1.csv"
football_data_2019_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2018-2019_ligue1.csv"
football_data_2020_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2019-2020_ligue1.csv"
football_data_2021_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2020-2021_ligue1.csv"
football_data_2022_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2021-2022_ligue1.csv"
football_data_2023_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2022-2023_ligue1.csv"
football_data_2024_adr="C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/raw/ligue1/matches/Football_data/Football-data.co.uk_2023-2024_ligue1.csv"

# --------------------------------------------------------------
# Convert CSV into DataFrame
# --------------------------------------------------------------
def read_data(files, Footy_or_Football):
    """
        Taking in entry the paths of the csv files, it outputs a DataFrame that is the concatenation of all our data.
        During this process the function converts 'date_GMT' or 'Date' column do datetime format, delete the raws of matchs definitivly canceled or not played yet (only for footy stats datasets as for football-data the canceled matchs do not appear).

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
    dataset = pd.concat(map(pd.read_csv, files), ignore_index=True)
    
    # Working with datetimes
    if Footy_or_Football == 'footy':
        dataset["date_GMT"]=dataset["date_GMT"].apply(lambda x: parse(x))
    else:
        for index, row in dataset.iterrows():
            if len(str(row['Date'])) == 8:
                dataset.at[index,'Date'] = pd.to_datetime(dataset.at[index,'Date'], format='%d/%m/%y')
                dataset.at[index,'Date'] = pd.to_datetime(dataset.at[index,'Date'], format='mixed')

            elif len(str(row['Date'])) == 10:
                dataset.at[index,'Date'] = pd.to_datetime(dataset.at[index,'Date'], format='%d/%m/%Y')
        dataset['Date'] = pd.to_datetime(dataset['Date'])
       

    # Deleting canceled matches lines
    if Footy_or_Football == 'footy':
        #Suppresession des lignes où le match a été annulé (covid) u pas encore joué. These matchs do not appear into Footaball-Data datasets
        rows_to_keep = dataset["status"] == "complete"

        dataset = dataset[rows_to_keep].reset_index(drop=True)
        
    return(dataset)


# --------------------------------------------------------------
# Define a function to load the dataframe saved at data/
# --------------------------------------------------------------

def load_data(seasons_present_in_df_info, file_name_0):
    """ 
        This function imports/loads a dataframe in data/ directory.
        It says what are the seasons represented in this dataframe if asked.
        
        Args:
        seasons_present_in_df_info (bool): Whether to print seasons info.
        file_name_0 (str): The name of the directory in /data where the dataset we look for is saved, followed by the name of the dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
    """
    
    dataset_location_path = f'C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/{file_name_0}.pkl'
 
    # Importation of dataset
    dataset = pd.read_pickle(dataset_location_path)
    
    if seasons_present_in_df_info == True:
        try:
             #Make out what are the seasons represented in the footy dataframe
            seasons_in_dataframe = dataset['date_GMT'].dt.year.unique()
        except KeyError:  # Assuming the error would be an AttributeError if 'date_GMT' does not exist
            seasons_in_dataframe = dataset['Date'].dt.year.unique()
            
        # Remove the holdest year which is only the beginning of the first season
        min_value = min(seasons_in_dataframe)
        seasons_in_dataframe = seasons_in_dataframe[seasons_in_dataframe != min_value]
        print(f"The {file_name_0} dataframe contains matchs of the seasons: ", seasons_in_dataframe)

    return dataset

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

def save_dataframe(dataset_0, file_name_0):
    """ 
        This function saves a dataframe to a specified location in the data/ directory. It deletes the old file at this location of the same name, if it exists, and compares the old and new dataframes (returns a string saying wether they are equal or not)

    
    Args:
        dataset_0 (Dataframe): the dataframe we want to save
        
        file_name_0 (str): the name of the directory in /data where saving the dataset, followed by the name we want to give to the file to save (the name we want to see appear in interim directory) ex: 'interim/feat_engineered_ds'.
        
    Returns:
        None
    """

    # Define the absolute path for the dataset destination
    dataset_destination_path = f"C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/{file_name_0}.pkl"
        
    
    # Load the old dataframe if it exists to compare
    try:
        old_dataframe = pd.read_pickle(dataset_destination_path)
        same_data = old_dataframe.equals(dataset_0)
        if same_data:
            print(f"The dataframes are the same for both old and new {file_name_0}")
        else:
            print(f"The dataframes are NOT the same for both old and new {file_name_0}")
    except FileNotFoundError:
        print(f"No {file_name_0} existing file to compare; treating as new dataset.")
    
    # Delete the old file if it exists
    try:
        os.remove(dataset_destination_path)
        print(f"Successfully deleted the old file:               {file_name_0}")
    except FileNotFoundError:
        print(f"No old file to delete at:              {file_name_0}")
    except Exception as e:
        print(f"An error occurred while deleting the old file: {e}")
        
   # Save the new dataframe
    try:
        dataset_0.to_pickle(dataset_destination_path)
        print(f"Successfully saved the new dataframe:            {file_name_0}")
    except Exception as e:
        print(f"An error occurred while saving the new dataframe: {e}")
    
    #Put a line break before at the end of the text w've just printed
    print("\n")

