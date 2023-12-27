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
#VARIABLES                 V 
def nb_matchs_nb_victories(dico_col_rk, dataset_0):
    nb_matchs_traites=0
    rownb_last_season_match=0

    for i in constant_variables.seasons:
        
        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison et on determine df utile pour le calcul du nb de lignes par saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
        
        #On initialise les deux dico avec les noms des équipes
        dico_matchs_played = useful_functions.init_dico_with_names_teams(equipes)
        dico_victories_nb = useful_functions.init_dico_with_names_teams(equipes)
        
        rownb_last_season_match+=df.shape[0]

        for j in range(nb_matchs_traites,rownb_last_season_match):
            
            HT_name= dataset_0.iloc[j,4]
            AT_name= dataset_0.iloc[j,5]
            
            #On remplit les colonnes HT_played_matchs_nb et AT_played_matchs_nb
            dataset_0.iloc[j,dico_col_rk['rg_HTPMN']] = dico_matchs_played[HT_name]
            dataset_0.iloc[j,dico_col_rk['rg_ATPMN']] = dico_matchs_played[AT_name]
            dataset_0.iloc[j,dico_col_rk['rg_HTVN']] = dico_victories_nb[HT_name]
            dataset_0.iloc[j,dico_col_rk['rg_ATVN']] = dico_victories_nb[AT_name]
            
            #On met a jour les dico avec les resultats du matchs de la ligne j
            dico_matchs_played[HT_name]+=1
            dico_matchs_played[AT_name]+=1
            
            if dataset_0.iloc[j, dico_col_rk['rg_RH']] == 1:
                dico_victories_nb[HT_name]+=1
            if dataset_0.iloc[j, dico_col_rk['rg_RA']] == 1:
                dico_victories_nb[AT_name]+=1
            
        nb_matchs_traites+=df.shape[0]
    
    return dataset_0


#Calculation and manipulation of VICTORY (pm, sbos):
#VARIABLE                  X
#PER MATCH AVG             V 
#PER MATCH AVG HT/AT DIFF  V
def victories_per_match_AVG_and_DIFF(dataset_0):
    #PER MATCH AVG VICTORY
    dataset_0["HT_avg_victory_pm"]=dataset_0["HT_victories_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_victory_pm"]=dataset_0["AT_victories_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    #Vérifié Vite fait

    #PER MATCH AVG HT/AT DIFF VICTORY
    dataset_0["Diff_HT_avg_victory_pm"] = dataset_0["HT_avg_victory_pm"] - dataset_0["AT_avg_victory_pm"]
    dataset_0["Diff_AT_avg_victory_pm"] = -dataset_0["Diff_HT_avg_victory_pm"]
    
    return dataset_0



#(POINTS COLLECTED x (1 + ADVERSARY AVG VICTORY PER MATCH)) (pm, sbos):
#VARIABLE                     V
#PER MATCH AVG                V
#PER MATCH AVG HT/AT DIFF     V
def points_pm_ponderated_by_adversary_perf(dico_col_rk, dataset_0):
    nb_matchs_traites=0
    rownb_last_season_match=0

    for i in (constant_variables.seasons):
        
        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison et on determine df utile pour le calcul du nb de lignes par saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
        #On initialise les deux dico avec les noms des équipes
        dico_points = useful_functions.init_dico_with_names_teams(equipes)
        
        rownb_last_season_match+=df.shape[0]

        for j in range(nb_matchs_traites, rownb_last_season_match):
            
            HT_name= dataset_0.iloc[j,4]
            AT_name= dataset_0.iloc[j,5]
            
            #On remplit les colonnes du DF
            dataset_0.iloc[j,dico_col_rk['rg_HTPPBAP']] = float(dico_points[HT_name])
            dataset_0.iloc[j,dico_col_rk['rg_ATPPBAP']] = float(dico_points[AT_name])
            
            #On récupère dans les variables res_ht et res_at les resultats des deux equipes
            res_ht = dataset_0.iloc[j,dico_col_rk['rg_RH']]
            res_at = dataset_0.iloc[j,dico_col_rk['rg_RA']]
            
            #On met a jour le dico avec les resultats du match de la ligne j
            dico_points[HT_name] += res_ht * 3 * (1 + dataset_0.iloc[j,dico_col_rk['rg_ATavgVpm']])
            dico_points[AT_name] += res_at * 3 * (1 + dataset_0.iloc[j,dico_col_rk['rg_HTavgVpm']])
            
            if res_ht == res_at and res_ht == 0 :
                dico_points[HT_name] += (1 + dataset_0.iloc[j,dico_col_rk['rg_ATavgVpm']])
                dico_points[AT_name] += (1 +dataset_0.iloc[j,dico_col_rk['rg_HTavgVpm']])
            
        nb_matchs_traites+=df.shape[0]

        
    #PER MTACH AVG
    dataset_0["HT_avg_pm_points_ponderated_by_adversary_perf"] = dataset_0["HT_points_ponderated_by_adversary_perf"] / (dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_pm_points_ponderated_by_adversary_perf"] = dataset_0["AT_points_ponderated_by_adversary_perf"] / (dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))  
        
    #PER MATCH HT/AT DIFF et on la place dans dataset et dataset_useful_features
    dataset_0["HT_Diff_points_ponderated_by_adversary_perf"] = dataset_0["HT_avg_pm_points_ponderated_by_adversary_perf"] - dataset_0["AT_avg_pm_points_ponderated_by_adversary_perf"]
    dataset_0["AT_Diff_points_ponderated_by_adversary_perf"] = -dataset_0["HT_Diff_points_ponderated_by_adversary_perf"]
    
    return dataset_0