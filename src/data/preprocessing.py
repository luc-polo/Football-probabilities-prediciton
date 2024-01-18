"""This module contains all the functions executing steps of preprocessing, implying data manipulation (not observation). That's mostly used into V) part
"""




import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data import constant_variables



# --------------------------------------------------------------
#  Getting formatted X, Y and splitting it into train, valid, test sets (used in 'V)4)')
# --------------------------------------------------------------
def formatting_splitting(dataset_restricted_0, col_to_delete_list, train_proportion, test_proportion, dataset_0):
    """  
    This function selects the data from dataset_restricted_0, removes rows where the nb of matchs played by teams is inferior to min_played_matchs_nb, removes column(s) of feature(s) we don't want to keep in our dataset (if there are), returns separated features and labels.
    Then it splits our data into train, valid and test sets and convert them into dataframes. It returns X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ready to be given tou our model and the calibrator.
    
    Args:  
        dataset_restricted_0 (DataFrame): Restricted dataset obtained in III)3)
        
        col_to_delete_list (list): (If there are) List of columns names we want to delete from dataset because we don't want the model to use it.
        
        train_proportion (int)
        
        test_proportion (int)
        
        dataset_0 (DataFrame): The dataset containing the data.
    
    Returns:
        Tuple: (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    """
    
    #Definition of Y
    #On  selectionne uniquement les lignes où le nb de match joués > min_played_matchs_nb (défini dans 'Definition of restricted datasets...')
    col1 = dataset_0[dataset_0['HT_played_matchs_nb']>constant_variables.min_played_matchs_nb]['RH']
    col2 = dataset_0[dataset_0['AT_played_matchs_nb']>constant_variables.min_played_matchs_nb]['RA']
    Df_concatenated_target_column = pd.concat([col1, col2], axis=0, ignore_index=True)
    #On convertit les deux colonnes concaténées en Dataframe
    Df_concatenated_target_column = pd.DataFrame(Df_concatenated_target_column, columns = ["Result"])

    (X,Y) = dataset_restricted_0, Df_concatenated_target_column
    
    #Supprimer les colonnes demandées
    for col in col_to_delete_list:
        X=X.drop(col, axis=1)

    
    # Convertir X et Y en tableaux NumPy pour les conformer au type de données pris en entrée par la fonction train_test_split
    X_np_values= X.values
    Y_np_values = Y.values
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_np_values, Y_np_values, test_size=(1-train_proportion), random_state=42, shuffle = True, stratify = Y_np_values)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=(test_proportion/(1-train_proportion)), random_state=42, shuffle = True, stratify = Y_temp)



    #Convertir X_train, X_test, Y_train, Y_test, X_valid, Y_valid en Dataframes
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    Y_train = pd.DataFrame(Y_train, columns=Y.columns)
    Y_test = pd.DataFrame(Y_test, columns=Y.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    Y_valid = pd.DataFrame(Y_valid, columns=Y.columns)
    
    return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)


    

