
import sys
import pandas as pd


# modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data import constant_variables




def formatting_splitting(dtaset_0):
    #Definition of Y
    #On  selectionne uniquement les lignes où le nb de match joués > min_played_matchs_nb (défini dans 'Definition of restricted datasets...')
    col1 = dataset_0[dataset_0['HT_played_matchs_nb']>min_played_matchs_nb]['RH']
    col2 = dataset_0[dataset_0['AT_played_matchs_nb']>min_played_matchs_nb]['RA']
    Df_concatenated_target_column = pd.concat([col1, col2], axis=0, ignore_index=True)
    #On convertit les deux colonnes concaténées en Dataframe
    Df_concatenated_target_column = pd.DataFrame(Df_concatenated_target_column, columns = ["Result"])

    (X,Y) = concat_restricted_ds_2, Df_concatenated_target_column


    #Supprimer la colonne Avg_xg si on utilise des données d'avant 2016
    def supp_avg_xg(X_0):
        X=X_0.drop('Avg_xg', axis=1)
        return(X)

    X = supp_avg_xg(X)


    # Convertir X et Y en tableaux NumPy pour les conformer au type de données pris en entrée par la fonction train_test_split
    X_np_values= X.values
    Y_np_values = Y.values


    # Split the data into training, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_np_values, Y_np_values, test_size=0.59, random_state=42, shuffle = True, stratify = Y_np_values)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, shuffle = True, stratify = Y_temp)



    #Convertir X_train, X_test, Y_train, Y_test, X_valid, Y_valid en Dataframes
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    Y_train = pd.DataFrame(Y_train, columns=Y.columns)
    Y_test = pd.DataFrame(Y_test, columns=Y.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    Y_valid = pd.DataFrame(Y_valid, columns=Y.columns)