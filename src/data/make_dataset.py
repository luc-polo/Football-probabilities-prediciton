"""
This module contains the function that creates the raw data dataframe.
"""

import pandas as pd
from dateutil.parser import parse
import sys
import os



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




files = [adr1, adr2, adr3, adr4, adr5, adr6, adr7, adr8, adr9]





# --------------------------------------------------------------
# Convert CSV into DataFrame
# --------------------------------------------------------------
def read_data(files):
    """
        Taking in entry the paths of the csv files, it outputs a DataFrame that is the concatenation of all our data.
        During this process the function converts 'date_GMT' column do datetime dormat, delete the raws of matchs definitivly.

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
# Export dataset
# --------------------------------------------------------------

def save_dataframe_into_data_interim(dataset_0):
    """ 
        This function saves the dataframe created with read_data() in data/interim. We just have to input the dataframe to save. The function deletes the old file at this location
    
    Args:
        dataset_0 (Dataframe): the dataframe we want to save
        
    Returns:
        None
    """

    # Define the absolute path of the dataset destination, which is also the path of the actual doc located there that we will delete
    dataset_destination_path = "C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/interim/data_interim.pkl"
    
    # We delete the previous file
    try: 
        os.remove(dataset_destination_path)
        print("The old 'data_interim.pkl' file was well deleted")
    except Exception as e:
        print(f"An error occurred while deleting the old datafram file: {e}")
        print("The old 'data_interim.pkl' file was not removed (probably because it does not exist in this location).")
        

    # Export the dataset to the path constructed
    try: 
        dataset_0.to_pickle(dataset_destination_path)
        print("The new 'data_interim.pkl' file was well saved")
    except Exception as e:
        print(f"An error occurred while saving the new dataframe: {e}")
        print("The 'data_interim.pkl' was not saved.")

# --------------------------------------------------------------
# Define a function to import the dataframe in other modules
# --------------------------------------------------------------

def load_data():
    """ 
    This fucntion imports the DataFrame we will name dataset, from data/interim/data_interim.
    It returns the dataset imported
    """
    
    daframe_location_path = 'C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/interim/data_interim.pkl'
    # Importation of dataset
    dataset = pd.read_pickle(daframe_location_path)
    
    return dataset