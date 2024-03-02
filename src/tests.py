import unittest
import pandas as pd
import os


# Test if the new dataset generated is different from the last one and give info about the modifications

def test_changes_on_dataset(dataset_0, dataset_0_name):
    """ Recap the alterations/changes in the main dataset (dataset_27 up to now) caused by modifications in the computation method of features or any minor adjustments.
    It details the features, seasons and teams names affected by the changes.

    Args:
        dataset_0 (_type_): The dataset in which we want to identify changes.
        dataset_0_name (_type_): The name of this dataset in fotmat str ex: 'dataset_27'
    """
    
    # Define the absolute path where we will save the dataset, which is also the location of the old one we will delete
    dataset_destination_path = "C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/data/processed/full_dataset.pkl"
    
    #We compare if the old dataset and the new one are exactly the same
    
    # Importation of dataset
    old_dataframe = pd.read_pickle(dataset_destination_path)
    
    if old_dataframe.equals(dataset_0):
        print(f"The old {dataset_0_name}, and the new one ARE the same \n")
    
    else:
        print("The old {dataset_0_name}_data_interim.pkl file, and the new one ARE NOT the same")
        print('\n')
        

        # Get columns where differences occur
        changed_columns = []
        for col in dataset_0.columns:
            if not dataset_0[col].equals(old_dataframe[col]):
                    changed_columns.append(col)
        print('The columns that are concerned by changes are:', changed_columns)
        
        for col in changed_columns:
            #we ask the user for each col if it's an HT or AT col
            HT_or_AT = input(input(f"Is {col} an HT or AT stat? ('H' for HT and 'A' for AT)"))
            print(col)
            #Get rows with changes in this co
            changed_rows = dataset_0.loc[dataset_0[col] != old_dataframe[col]]

            #Get the seasons with changes in this col
            changed_seasons = changed_rows['Season_year'].unique()

            print("The seasons during which we observe changes in", col, "are:", changed_seasons)
            for season in changed_seasons:
                #Get raws with changes in this col during this season
                changed_rows_in_this_season = changed_rows[changed_rows['Season_year'] == season]
                
                #Indentify the teams concerned by changes
                if HT_or_AT == 'H':
                    changed_teams = changed_rows_in_this_season['home_team_name'].unique()
                elif HT_or_AT == 'A':
                    changed_teams = changed_rows_in_this_season['away_team_name'].unique()
                
                print("The teams that observe changes in", col, "during the season", season, "are:", changed_teams)
    
    #We delete the old dataset
    # We delete the old file 
    try: 
        os.remove(dataset_destination_path)
        print(f"The old '{dataset_0_name}' file was well deleted")
    except Exception as e:
        print(f"An error occurred while deleting the old {dataset_0_name} dataframe : {e}")
        print(f"The old '{dataset_0_name}' file was not removed (probably because it does not exist in this location).")
        

    # We save the new dataset
    try: 
        dataset_0.to_pickle(dataset_destination_path)
        print(f"The new '{dataset_0_name}' file was well saved \n")
    except Exception as e:
        print(f"An error occurred while saving the new {dataset_0_name} dataframe: {e}")
        print("The '{footy_or_football}' was not saved. \n")
    