""" 
This module is made to create the new features we want to use for our model, or at least to test their relevancy.
"""

import numpy as np
import sys

# modify the sys.path list to include the path to the data directory that contains the modules that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data import constant_variables
import useful_functions





# Calculation of MATCHS RESULTS

#Ces deux colonnes contiendront un "1" (pour signaler une victoire) ou un "0" (pour signaler un nul ou une défaite)fded
# Colonne "RH" qui vaut 1 lorsque "home_score" > "away_score" sinon 0
def matchs_results(dataset_0):
    dataset_0["RH"] = np.where(dataset_0["home_team_goal_count"] > dataset_0["away_team_goal_count"], 1, 0)
    # Colonne "RA" qui vaut 1 lorsque "away_score" > "home_score" sinon 0
    dataset_0["RA"] = np.where(dataset_0["away_team_goal_count"] > dataset_0["home_team_goal_count"], 1, 0)
    
    return dataset_0


# Calculation of Home or Away status of the teams
def home_away_status(dataset_0):
    L,l = dataset_0.shape
    dataset_0["HT_H_A_status"] = [1 for i in range(L)]
    dataset_0["AT_H_A_status"] = [0 for i in range(L)]
    
    return dataset_0




#Calculation and manipulation of NB DE MATCHS et NB DE VICTOIRES (pm, sbos)
#VARIABLES  V 

def nb_matchs_nb_victories(dico_col_rk, dataset_0):
    nb_matchs_traites=0
    rownb_last_season_match=0

    for i in constant_variables.seasons:
        
        #On créer "equipes" qui contient les noms de toutes les équipes du championnat durant une saison et on determine df utile pour le calcul du nb de lignes par saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
        
        #On initialise les deux dico avec les noms des équipes
        dico_matchs_played = useful_functions.init_dico_with_names_teams(equipes)
        dico_victories_nb = useful_functions.init_dico_with_names_teams(equipes)
        
        rownb_last_season_match+=df.shape[0]

        for j in range(nb_matchs_traites,rownb_last_season_match):
            
            HT_name= dataset_0.iloc[j,4]
            AT_name= dataset_0.iloc[j,5]
            
            #On remplit les colonnes HT_played_matchs_nb et AT_played_matchs_nb
            dataset_0.iloc[j,dico_col_rk.rg_HTPMN] = dico_matchs_played[HT_name]
            dataset_0.iloc[j,dico_col_rk.rg_ATPMN] = dico_matchs_played[AT_name]
            dataset_0.iloc[j,dico_col_rk.rg_HTVN] = dico_victories_nb[HT_name]
            dataset_0.iloc[j,dico_col_rk.rg_ATVN] = dico_victories_nb[AT_name]
            
            #On met a jour les dico avec les resultats du matchs de la ligne j
            dico_matchs_played[HT_name]+=1
            dico_matchs_played[AT_name]+=1
            
            if dataset_0.iloc[j, dico_col_rk.rg_RH] == 1:
                dico_victories_nb[HT_name]+=1
            if dataset_0.iloc[j, dico_col_rk.rg_RA] == 1:
                dico_victories_nb[AT_name]+=1
            
        nb_matchs_traites+=df.shape[0]
    
    return dataset_0