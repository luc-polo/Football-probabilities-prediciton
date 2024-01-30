""" 
This module is made to create the new features we want to use for our model, or at least to test their relevancy.
"""

import numpy as np
import sys
from dateutil.relativedelta import relativedelta
import pandas as pd


# modify the sys.path list to include the path to the data directory that contains the modules that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables
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


#Calculation of the season year
#VARIABLE
def season_year(dataset_0):
    dataset_0['Season_year'].astype(str)
    for season in constant_variables.seasons:
        start_date = season - relativedelta(years=1)
        end_date = season
        dataset_0.loc[(start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date), 'Season_year'] = '2019'
        
        
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
    
    #on change le type des colonnes car cela a posé problème
    dataset_0["HT_points_ponderated_by_adversary_perf"] = dataset_0["HT_points_ponderated_by_adversary_perf"].astype('float64')
    dataset_0["AT_points_ponderated_by_adversary_perf"] = dataset_0["AT_points_ponderated_by_adversary_perf"].astype('float64')

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



#Calculation and manipulation of NB OF SCORED GOALS and NB CONCEEDED GOALS (pm, sbos)
#VARRIABLE                    V
def scored_and_conceeded_goals(dico_col_rk, dataset_0):
    #sert pour la boucle qui va permettre de remplir les dictionnaires "dico_goals_scored" et "dico_goals_conceeded"
    nb_matchs_traites=0
    rownb_last_season_match=0
    

    for i in (constant_variables.seasons):
    #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
    #On initialise les deux dico avec les noms des équipes
        dico_goals_scored = useful_functions.init_dico_with_names_teams(equipes)
        dico_goals_conceeded = useful_functions.init_dico_with_names_teams(equipes)
        rownb_last_season_match+=df.shape[0]
    #On remplit les deux dicos qui contiendront le nom des 20 (ou 18) equipes et le nombre de buts marqués et encaissés.
    #On remplit également 4 colonnes buts marqués/encaissés par la Home et Away team (pre match).

        for j in range(nb_matchs_traites,rownb_last_season_match):
    #On remplit la colonne "buts marqués pre-match par la HT" et on met a jour le dictionnaire de buts marqués pour la HT avec le résultat du match
            dataset_0.iloc[j,dico_col_rk['rg_SGHTPM']]=dico_goals_scored[dataset_0.iloc[j,4]]
            dico_goals_scored[dataset_0.iloc[j,4]]+=dataset_0.iloc[j,12]
            
    #On remplit la colonne "buts encaissés pre-match par la HT" et on met a jour le dictionnaire de buts encaissés pour la HT avec le résultat du match
            dataset_0.iloc[j,dico_col_rk['rg_CGHTPM']] = dico_goals_conceeded[dataset_0.iloc[j,4]]
            dico_goals_conceeded[dataset_0.iloc[j,4]]+=dataset_0.iloc[j,13]
            
    #On remplit la colonne "buts marqués pre-match par la AT" et on met a jour le dictionnaire de buts marqués pour la AT avec le résultat du match
            dataset_0.iloc[j,dico_col_rk['rg_SGATPM']] = dico_goals_scored[dataset_0.iloc[j,5]]
            dico_goals_scored[dataset_0.iloc[j,5]]+=dataset_0.iloc[j,13]
            
    #On remplit la colonne "buts encaissés pre-match par la AT" et on met a jour le dictionnaire de buts encaissés pour la AT avec le résultat du match
            dataset_0.iloc[j,dico_col_rk['rg_CGATPM']] = dico_goals_conceeded[dataset_0.iloc[j,5]]
            dico_goals_conceeded[dataset_0.iloc[j,5]]+=dataset_0.iloc[j,12]
        
        nb_matchs_traites+=df.shape[0]
            
    return dataset_0

#GOAL DIFFERENCE (pm, sbos): 
#VARIABLE                     V
#PER MATCH AVG                V
#PER MATCH AVG HT/AT DIFF     V
def goal_difference(dataset_0):
    #GOAL DIFFERENCE
    dataset_0["goal_diff_HT_PM"] = dataset_0["scored_goals_HT_PM"] - dataset_0["conceeded_goals_HT_PM"]
    dataset_0["goal_diff_AT_PM"] = dataset_0["scored_goals_AT_PM"] - dataset_0["conceeded_goals_AT_PM"]
    #OK (12/09/23)

    #PER MATCH AVG GOAL DIFF
    dataset_0["HT_avg_goal_diff_pm"] = dataset_0["goal_diff_HT_PM"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_goal_diff_pm"] = dataset_0["goal_diff_AT_PM"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))

    #PER MATCH AVG HT/AT DIFF GOAL DIFF:
    dataset_0["Diff_HT_goal_diff_pm"] = dataset_0["HT_avg_goal_diff_pm"] - dataset_0["AT_avg_goal_diff_pm"]
    dataset_0["Diff_AT_goal_diff_pm"] = - dataset_0["Diff_HT_goal_diff_pm"]
    
    return dataset_0

#SCORED GOALS / CONCEEDED GOALS (pm, sbos):
#VARIABLE                    V
#PER MATCH AVG               X 
#HT/AT DIFF                  V
def scored_conceeded_goals_ratio(dataset_0):
    #(SCORED GOALS / CONCEEDED GOALS)
    dataset_0["HT_avg_scored_g_conceedded_g_ratio"] = dataset_0["scored_goals_HT_PM"]/(dataset_0["conceeded_goals_HT_PM"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_scored_g_conceedded_g_ratio"] = dataset_0["scored_goals_AT_PM"]/(dataset_0["conceeded_goals_AT_PM"].apply(useful_functions.un_ou_x))
    #OK
    #Vérifié vitfait sur la saison 2016-2017

    #(SCORED GOALS / CONCEEDED GOALS) HT/AT DIFF
    dataset_0["Diff_HT_avg_scored_g_conceedded_g_ratio"] = dataset_0["HT_avg_scored_g_conceedded_g_ratio"] - dataset_0["AT_avg_scored_g_conceedded_g_ratio"] 
    dataset_0["Diff_AT_avg_scored_g_conceedded_g_ratio"] = dataset_0["AT_avg_scored_g_conceedded_g_ratio"] - dataset_0["HT_avg_scored_g_conceedded_g_ratio"]
    
    return dataset_0

#POINTS NB (pm, sbos):
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def points_nb( dico_col_rk,dataset_0):
    #sert pour la boucle qui va permettre de remplir le dictionnaire "points_nb_dico"
    nb_matchs_trates=0
    rownb_last_season_match=0
    
    for i in (constant_variables.seasons):
        points_nb_dico={}

        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison 
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)

        #On initialise le dico avec les noms des équipes
        points_nb_dico = useful_functions.init_dico_with_names_teams(equipes)
            
        rownb_last_season_match+=df.shape[0]
        
    #On remplit le dico qui contient le nom des 20 (ou 18) equipes et le nombre de points prematch des équipes.
    #On remplit également les 2 colonnes nb points pre-match de la Home et Away team (pre match).
        
        for j in range(nb_matchs_trates,rownb_last_season_match):
            
    #On remplit les colonnes "Prematch_HT_PN", "Prematch_AT_PN" et on met a jour le dictionnaire
            dataset_0.iloc[j,dico_col_rk['rg_PHTPN']]=points_nb_dico[dataset_0.iloc[j,4]]
            dataset_0.iloc[j,dico_col_rk['rg_PATPN']] = points_nb_dico[dataset_0.iloc[j,5]]
            if dataset_0.iloc[j, dico_col_rk['rg_RH']] == 0 and dataset_0.iloc[j,dico_col_rk['rg_RA']] == 0:
                points_nb_dico[dataset_0.iloc[j,4]]+=1
                points_nb_dico[dataset_0.iloc[j,5]]+=1
            else:
                points_nb_dico[dataset_0.iloc[j,4]]+= 3*dataset_0.iloc[j,dico_col_rk['rg_RH']]
                points_nb_dico[dataset_0.iloc[j,5]]+= 3*dataset_0.iloc[j,dico_col_rk['rg_RA']]

        
        nb_matchs_trates+=df.shape[0]

    #Le programme ci-dessus a été vérifié pour la saison 2016-2017. Les points pre-match sont correctement calculés jusqu'à la dèrnière journée de championnat. La seule anomalie relevée est avec un match (Bastia-Lyon) qui a été suspendu et gagné sur tapis vert par Lyon. Or dans le fichier Excel il est marqué "complete" avec comme resultat: 0-0. L'algo a donc rajouté un point a chaque team. Il faut donc etre vigilant avec les matchs annulés et perdus sur tapis vert

    #OK (SIMPLEMENT DISCUSSION POSSIBLE SUR LE CALCUL DES POINTS ISSUS DES MATCHS PERDUS SUR TAPIS VERT)

    #PER MATCH AVG POINTS NB
    dataset_0["HT_avg_collected_points_pm"]=dataset_0["Prematch_HT_PN"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_collected_points_pm"]=dataset_0["Prematch_AT_PN"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    #OK
    #Vérifié vitfait

    #HT/AT DIFF PER MATCH AVG POINTS NB
    dataset_0["Diff_pnt_HT_AT_ratio"] = (dataset_0["Prematch_HT_PN"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["Prematch_AT_PN"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["Diff_pnt_AT_HT_ratio"] = (dataset_0["Prematch_AT_PN"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["Prematch_HT_PN"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#RANKING (pm, sbos) :
#VARIABLE                    V
#PER MATCH AVG               X
#HT/AT DIFF                  V
def ranking(dico_col_rk, dataset_0):
    #Ce  classement classe les teams selon leur nombre de point et goal difference, mais ne prend pas en compte le nb de buts marqués en cas d'égalité de goal average et de points
    L,largeur = dataset_0.shape


    for i in (constant_variables.seasons):

        start_date = i - relativedelta(years=1)
        end_date = i
        
        for w in range(1,constant_variables.nb_championship_weeks + 1):
            pnt_list=[]
            ranking=[]
            goal_diff_list=[]
            #On sélectionne les indices des lignes qui ont Game Week = w et qui ont une date qui est compris entre start_date and end_date.
            Indices=np.where((dataset_0["Game Week"]==w) & (start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date))[0]
            
            if Indices.any():
                #On supprime les matchs qui ont été reportés:
                Indices = [y for y in Indices if y <= (Indices[0] + ((constant_variables.nb_teams / 2) - 1))]

                #On fait une première boucle qui va permettre de classer les équipes avant le début de la w ème journée:
                for y in Indices:

                    #On classe les equipes dans les 3 listes (ranking, pnt_list , goal_diff_list) qui vont constituer le classement de la w ème journée:
                    #HOME TEAM
                    useful_functions.classement_team(y, ranking, pnt_list , goal_diff_list, "home", dico_col_rk, dataset_0)

                    #AWAY TEAM
                    useful_functions.classement_team(y, ranking, pnt_list , goal_diff_list, "away", dico_col_rk, dataset_0)
                    
                    
                #Si il manque une ou plusieurs équipes au classement on va chercher leur prochain match pour avoir leurs stats pre-match et les incorporer au classement
                useful_functions.ajout_missing_teams_ranking( y, ranking, pnt_list, goal_diff_list, dico_col_rk, dataset_0)
                
                #On remplit pour chaque match de la w ème journée les classements prematch des HT et AT:
                useful_functions.fill_dataset_with_teams_rank(Indices, ranking, dico_col_rk, dataset_0)            
                
                
                #Si le ou les matchs qui précèdent (chronologiquement et donc aussi dans l'ordre des lignes du dataset) le premier de la journée w,
                #sont des matchs reportés, alors on va classer ces equipes dans le classement réalisé précedemment pour la journée w, en ayant 
                #préalablement retiré les equipes en question du classement
                nb_line_last_match_of_week = min(Indices)

                if nb_line_last_match_of_week != 0 :
                    l= nb_line_last_match_of_week - 1

                    while dataset_0.at[l, "Game Week"] < (w-1) and (start_date < dataset_0.at[l,"date_GMT"]) and (dataset_0.at[l,"date_GMT"] <= end_date) and (dataset_0.iloc[l,dico_col_rk['rg_HTWR']] == 0):
                        
                        """print(l, "On ratrappe la journée:",dataset.at[l, "Game Week"], dataset.iloc[l, 4],"-",dataset.iloc[l, 5])"""
                        
                        #On supprime les equipes du match reporté en question, des classements de la w eme journee
                        useful_functions.delete_postponned_match_teams_from_ranking(ranking, pnt_list, goal_diff_list, l, dataset_0)

                        #On classe les equipes du match dans les classements de la w eme journee
                        useful_functions.classement_team(l, ranking, pnt_list, goal_diff_list, "home", dico_col_rk, dataset_0)
                        useful_functions.classement_team(l, ranking, pnt_list, goal_diff_list, "away", dico_col_rk, dataset_0)
                        
                        #On remplit pour chaque match de la w ème journée les classements prematch des HT et AT:
                        dataset_0.iloc[l, dico_col_rk['rg_HTWR']]= ranking.index(dataset_0.iloc[l,4]) + 1
                        dataset_0.iloc[l, dico_col_rk['rg_ATWR']]= ranking.index(dataset_0.iloc[l,5]) + 1
                        
                        #Pour Tester:
                        """print("le ranking avant le match rattrapé est:")
                        for b in range(len(ranking)):
                            print(ranking[b],":", pnt_list[b])
                        print("rank ht:",dataset.iloc[l, rg_HTWR])
                        print("rank at:",dataset.iloc[l, rg_ATWR])
                        print("\n")"""   
                        
                        l-=1

    #Le classement est OK et verifié  (25/09/23)

    #On calcule la  différence HT/AT pour le ranking et on la place dans dataset
    dataset_0["Diff_HT_ranking"]  = -(dataset_0["HT_week_ranking"] - dataset_0["AT_week_ranking"])
    dataset_0["Diff_AT_ranking"]  = -(dataset_0["AT_week_ranking"] - dataset_0["HT_week_ranking"])
    
    return dataset_0

#ANNUAL BUDGET:
#VARIABLE                    V
#PER MATCH AVG               X
#HT/AT DIFF                  V
def annnual_budget(dataset_0):
    
    #Dicos qui contiennent les équipes à chaque saison et leurs budgets
    budgets_2015={"PSG": 480, 'Toulouse' : 32, 'Marseille': 105, 'Rennes': 40, 'Nice': 40, 'Lyon':115, 'Bastia':22, 'Lille':65, 'Reims': 30, 'Lorient': 36, 'Caen':26, 'Nantes':32, 'Metz':28, 'Evian':28, 'Montpellier':40, 'St Etienne':50, 'Bordeaux':55, 'Guingamp':25, 'Monaco':160, 'Lens':38 }
    budgets_2016={"PSG": 490, 'Toulouse' : 30, 'Marseille': 120, 'Rennes': 43, 'Nice': 40, 'Lyon':170, 'Bastia':25, 'Lille':75, 'Reims': 31, 'Lorient': 30, 'Caen':26, 'Nantes':38, 'Troyes':23, 'Angers':24, 'Montpellier':42, 'St Etienne':68, 'Bordeaux':55, 'Guingamp':25, 'Monaco':140, 'Gazélec Ajaccio':25 }
    budgets_2017={"PSG": 503, 'Toulouse' : 37, 'Marseille': 100, 'Rennes': 50, 'Nice': 45, 'Lyon':235, 'Nancy':30, 'Lille':75, 'Metz': 30, 'Lorient': 36, 'Caen':29, 'Nantes':40, 'Dijon':26, 'Angers':29, 'Montpellier':42, 'St Etienne':70, 'Bordeaux':60, 'Guingamp':26, 'Monaco':145, 'Bastia':28 }
    budgets_2018={"PSG": 540, 'Toulouse' : 34, 'Marseille': 120, 'Rennes': 50, 'Nice': 45, 'Lyon':240, 'Amiens':25, 'Lille':90, 'Metz': 33, 'Strasbourg': 30, 'Caen':32, 'Nantes':45, 'Dijon':32, 'Angers':28, 'Montpellier':43.5, 'St Etienne':68, 'Bordeaux':65, 'Guingamp':26, 'Monaco':170, 'Troyes': 26 }
    budgets_2019={"PSG": 500, "Lyon": 285, "Monaco": 215, "Marseille": 150, "Lille": 90, "St Etienne": 74, "Bordeaux": 70, "Rennes": 68, "Nantes": 60, "Nice": 50, "Montpellier": 41.4, "Reims": 40, "Strasbourg": 37.5, "Amiens": 36, "Dijon": 35, "Toulouse": 35, "Caen": 34, "Angers": 30, "Guingamp": 30, "Nîmes": 20} 
    budgets_2020={"PSG": 637, "Lyon": 310, "Monaco": 220, "Marseille": 110, "Lille": 120, "St Etienne": 100, "Bordeaux": 70, "Rennes": 65, "Nantes": 70, "Nice": 50, "Montpellier": 40, "Reims": 45, "Strasbourg": 43, "Amiens": 30, "Dijon": 38, "Toulouse": 35, "Nîmes": 27, "Angers": 32, "Brest": 30, "Metz": 40} 
    budgets_2021={"PSG": 640, "Lyon": 285, "Monaco": 215, "Marseille": 140, "Lille": 147, "St Etienne": 95, "Bordeaux": 65, "Rennes": 105, "Nantes": 75, "Nice": 75, "Montpellier": 54.5, "Reims": 70, "Strasbourg": 50, "Lens": 46, "Dijon": 50, "Lorient": 45, "Nîmes": 40, "Angers": 45, "Brest": 35, "Metz": 50} 
    budgets_2022={"PSG": 500, "Lyon": 250, "Monaco": 215, "Marseille": 250, "Lille": 147, "St Etienne": 70, "Bordeaux": 112, "Rennes": 110, "Nantes": 65, "Nice": 90, "Montpellier": 44, "Reims": 60, "Strasbourg": 45, "Lens": 43, "Troyes": 30, "Lorient": 50, "Clermont": 20, "Angers": 41, "Brest": 32, "Metz": 50}
    budgets_2023={"PSG": 700, "Lyon": 250, "Monaco": 240, "Marseille": 250, "Lille": 100, "Auxerre": 32, "Ajaccio": 22, "Rennes": 90, "Nantes": 75, "Nice": 100, "Montpellier": 52, "Reims": 70, "Strasbourg": 45, "Lens": 62, "Troyes": 45, "Lorient": 50, "Clermont": 25, "Angers": 40, "Brest": 48, "Toulouse": 40}

    dicos_budgets=[budgets_2015, budgets_2016, budgets_2017, budgets_2018, budgets_2019, budgets_2020, budgets_2021, budgets_2022, budgets_2023]

    j=0
    
    # Convertir les colonnes en float64 avant d'assigner des valeurs, car j'ai eu des prblms
    dataset_0["annual budget of HT"] = dataset_0["annual budget of HT"].astype('float64')
    dataset_0["annual budget of AT"] = dataset_0["annual budget of AT"].astype('float64')
    
    for i in (constant_variables.seasons):
        
        #Permet de traiter uniquement les lignes correspondant à une saison:
        start_date = i - relativedelta(years=1)
        end_date = i
        
        #On sélectionne le dictionnaire budget associé à la saison traitée
        budgets=dicos_budgets[j]
        j+=1
        
        #Pour chaque equipe on remplit sur chaque ligne ou elle apparait son budget (ce pour les match à dom et à l'ext)
        for e in budgets:
            dataset_0.loc[(dataset_0["home_team_name"]== e) & (start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date),["annual budget of HT"]] = float(budgets[e])
            dataset_0.loc[(dataset_0["away_team_name"]== e) & (start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date),["annual budget of AT"]] = float(budgets[e])
                

    #TOUT EST OK (12/09/23)

    #HT/AT DIFF 
    #On la place dans dataset
    dataset_0["Diff_HT_annual_budget"] = dataset_0["annual budget of HT"] - dataset_0["annual budget of AT"]
    dataset_0["Diff_AT_annual_budget"] = dataset_0["annual budget of AT"] - dataset_0["annual budget of HT"]
    
    return dataset_0

#POINTS NB ON 1,3,5 LAST MATCHS
#VARIABLE                    V
#PER MATCH AVG               X
#HT/AT DIFF                  V
def points_nb_on_x_last_matchs(dico_col_rk, dataset_0):
    nb_matchs_trates=0
    rownb_last_match_of_season=0

    for i in (constant_variables.seasons):
        
        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
        
        rownb_last_match_of_season+=df.shape[0]
        
        #On initialise les dataframes ("DF_pn_5lm", "DF_pn_3lm", "DF_pn_1lm") avec les noms des équipes. IL contiendra le nombre de points pris lors 
        #des 1,3,5 derniers matchs par chaque équipe du championnat. Chaque colonne porte le nom d'une équipe.
        #La première ligne contient le nb de points pris lors du dernier match joué par les équipes, la deuxième ligne le
        #nb de points pris lors de l'avant derniern match joué...
        
        DF_pn_5lm = pd.DataFrame({x:[0 for i in range(5)] for x in equipes})
        DF_pn_3lm = pd.DataFrame({x:[0 for i in range(3)] for x in equipes})
        DF_pn_1lm = pd.DataFrame({x:[0] for x in equipes})
        
        #On met a jour DF_pn_5lm a chaque match de la saison et on remplit les colonnes Points_HT_5lm_PM et Points_AT_5lm_PM
        for j in range(nb_matchs_trates,rownb_last_match_of_season):

            nom_HT=dataset_0.iloc[j,4]
            nom_AT=dataset_0.iloc[j,5]
            
            #On remplit les colonnes "Points_HT_1lm_PM", "Points_AT_1lm_PM", "Points_HT_3lm_PM" ... :
            dataset_0.iloc[j,dico_col_rk['rg_PHT5LMPM']] = DF_pn_5lm[nom_HT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_PAT5LMPM']] = DF_pn_5lm[nom_AT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_PAT3LMPM']] = DF_pn_3lm[nom_AT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_PHT1LMPM']] = DF_pn_1lm[nom_HT]
            dataset_0.iloc[j,dico_col_rk['rg_PHT3LMPM']] = DF_pn_3lm[nom_HT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_PAT1LMPM']] = DF_pn_1lm[nom_AT]
            
            #On met à jour DF_pn_5lm
            #On commence par faire descendre d'une ligne les colonnes correspondantes à la HT et AT
            for k in range(4,0,-1):
                DF_pn_5lm.at[k,nom_HT] = DF_pn_5lm.at[k-1,nom_HT]
                DF_pn_5lm.at[k,nom_AT] = DF_pn_5lm.at[k-1,nom_AT]
            for k in range(2,0,-1):
                DF_pn_3lm.at[k,nom_HT] = DF_pn_3lm.at[k-1,nom_HT]
                DF_pn_3lm.at[k,nom_AT] = DF_pn_3lm.at[k-1,nom_AT]
                
            #Puis on remplit la première ligne du dataframe (ligne qui correspond au nb de points pris lors du dernier match joué) pour la HT et AT
            if dataset_0.iloc[j,dico_col_rk['rg_RH']] == dataset_0.iloc[j,dico_col_rk['rg_RA']] == 0:
                DF_pn_5lm.at[0,nom_HT]=1
                DF_pn_5lm.at[0,nom_AT]=1
                DF_pn_3lm.at[0,nom_HT]=1
                DF_pn_3lm.at[0,nom_AT]=1
                DF_pn_1lm.at[0,nom_HT]=1
                DF_pn_1lm.at[0,nom_AT]=1
            else:
                DF_pn_5lm.at[0,nom_HT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RH']]
                DF_pn_5lm.at[0,nom_AT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RA']]
                DF_pn_3lm.at[0,nom_HT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RH']]
                DF_pn_3lm.at[0,nom_AT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RA']]
                DF_pn_1lm.at[0,nom_HT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RH']]
                DF_pn_1lm.at[0,nom_AT]= 3*dataset_0.iloc[j,dico_col_rk['rg_RA']]
            
        nb_matchs_trates+=df.shape[0]


    #OK
    #(Vérifié rapidement en checkant DF_pn_5lm à la dèrnière journée de champ des saison 2021, 2017, 2016 et DF_pn_3lm, DF_pn_1lm pour la dernière journée de 2021)


    #HT/AT DIFF
    dataset_0["HT_Diff_Points_5lm"] = dataset_0["Points_HT_5lm_PM"] - dataset_0["Points_AT_5lm_PM"]
    dataset_0["AT_Diff_Points_5lm"] = -dataset_0["HT_Diff_Points_5lm"]
    dataset_0["HT_Diff_Points_3lm"] = dataset_0["Points_HT_3lm_PM"] - dataset_0["Points_AT_3lm_PM"]
    dataset_0["AT_Diff_Points_3lm"] = -dataset_0["HT_Diff_Points_3lm"]
    dataset_0["HT_Diff_Points_1lm"] = dataset_0["Points_HT_1lm_PM"] - dataset_0["Points_AT_1lm_PM"]
    dataset_0["AT_Diff_Points_1lm"] = -dataset_0["HT_Diff_Points_1lm"]
    
    return dataset_0

#GOAL DIFF ON 1,3,5 LAST MATCHS
#VARIABLE                    V
#PER MATCH AVG               X
#HT/AT DIFF                  V
def goal_diff_on_x_last_matchs(dico_col_rk, dataset_0):
    nb_matchs_trates=0
    rownb_last_match_of_season=0

    for i in (constant_variables.seasons):
        
        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)
        
        rownb_last_match_of_season+=df.shape[0]
        
        
        #On initialise les dataframes ("DF_gd_5lm", "DF_gd_3lm", "DF_gd_1lm") avec les noms des équipes. IL contiendra la goal diff sur 
        #les 1,3,5 derniers matchs par chaque équipe du championnat. Chaque colonne porte le nom d'une équipe.
        #La première ligne contient la goall diff sur le dernier match joué par les équipes, la deuxième ligne le
        #nb de points pris lors de l'avant derniern match joué...
        
        DF_gd_5lm = pd.DataFrame({x:[0 for i in range(5)] for x in equipes})
        DF_gd_3lm = pd.DataFrame({x:[0 for i in range(3)] for x in equipes})
        DF_gd_1lm = pd.DataFrame({x:[0] for x in equipes})

        #On met a jour DF_gd_xlm a chaque match de la saison et on remplit les colonnes GoalDiff_HT_xlm_PM et GoalDiff_AT_xlm_PM
        for j in range(nb_matchs_trates,rownb_last_match_of_season):

            nom_HT=dataset_0.iloc[j,4]
            nom_AT=dataset_0.iloc[j,5]
            
            #On remplit les colonnes GoalDiff_HT_xlm_PM et GoalDiff_AT_xlm_PM :
            dataset_0.iloc[j,dico_col_rk['rg_GDHT5LMPM']] = DF_gd_5lm[nom_HT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_GDAT5LMPM']] = DF_gd_5lm[nom_AT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_GDHT3LMPM']] = DF_gd_3lm[nom_HT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_GDAT3LMPM']] = DF_gd_3lm[nom_AT].sum()
            dataset_0.iloc[j,dico_col_rk['rg_GDHT1LMPM']] = DF_gd_1lm[nom_HT]
            dataset_0.iloc[j,dico_col_rk['rg_GDAT1LMPM']] = DF_gd_1lm[nom_AT]
            
            
            #On met à jour DF_gd_xlm
            #On commence par faire descendre d'une ligne les colonnes correspondantes à la HT et AT
            for k in range(4,0,-1):
                DF_gd_5lm.at[k,nom_HT] = DF_gd_5lm.at[k-1,nom_HT]
                DF_gd_5lm.at[k,nom_AT] = DF_gd_5lm.at[k-1,nom_AT]
            for k in range(2,0,-1):
                DF_gd_3lm.at[k,nom_HT] = DF_gd_3lm.at[k-1,nom_HT]
                DF_gd_3lm.at[k,nom_AT] = DF_gd_3lm.at[k-1,nom_AT]
                
            #Puis on remplit la première ligne du dataframe (ligne qui correspond à la Goal Diff du dernier match joué, c-a-d celui de la ligne j considérée) pour la HT et AT
            DF_gd_5lm.at[0,nom_HT] = dataset_0.iloc[j,12] - dataset_0.iloc[j,13]
            DF_gd_5lm.at[0,nom_AT] = dataset_0.iloc[j,13] - dataset_0.iloc[j,12]
            DF_gd_3lm.at[0,nom_HT] = dataset_0.iloc[j,12] - dataset_0.iloc[j,13]
            DF_gd_3lm.at[0,nom_AT] = dataset_0.iloc[j,13] - dataset_0.iloc[j,12]
            DF_gd_1lm.at[0,nom_HT] = dataset_0.iloc[j,12] - dataset_0.iloc[j,13]
            DF_gd_1lm.at[0,nom_AT] = dataset_0.iloc[j,13] - dataset_0.iloc[j,12]
        
        nb_matchs_trates+=df.shape[0]


    #Fini, vérifié


    #HT/AT DIFF
    dataset_0["HT_Diff_Goal_Diff_5lm"] = dataset_0["GoalDiff_HT_5lm_PM"] - dataset_0["GoalDiff_AT_5lm_PM"]
    dataset_0["AT_Diff_Goal_Diff_5lm"] = -dataset_0["HT_Diff_Goal_Diff_5lm"]
    dataset_0["HT_Diff_Goal_Diff_3lm"] = dataset_0["GoalDiff_HT_3lm_PM"] - dataset_0["GoalDiff_AT_3lm_PM"]
    dataset_0["AT_Diff_Goal_Diff_3lm"] = -dataset_0["HT_Diff_Goal_Diff_3lm"]
    dataset_0["HT_Diff_Goal_Diff_1lm"] = dataset_0["GoalDiff_HT_1lm_PM"] - dataset_0["GoalDiff_AT_1lm_PM"]
    dataset_0["AT_Diff_Goal_Diff_1lm"] = -dataset_0["HT_Diff_Goal_Diff_1lm"]
    
    return dataset_0

#RANKING ON 1,3,5 LAST MATCHS (pm)
#VARIABLE                    V
#PER MATCH AVG               X
#HT/AT DIFF                  X
def ranking_on_x_last_matchs(dico_col_rk_0, dataset_0):
    #Ce  classement classe les teams selon leur nombre de point et goal difference, mais ne prends pas en compte le nb de buts marqués 
    #en cas d'égalité de goal average et de points


    for i in (constant_variables.seasons):

        start_date = i - relativedelta(years=1)
        end_date = i
        
        for w in range(1,constant_variables.nb_championship_weeks + 1):
            
            
            #On sélectionne les indices des lignes qui ont Game Week = w et qui ont une date qui est comprise entre start_date and end_date.
            Indices=np.where((dataset_0["Game Week"]==w) & (start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date))[0]
            
            if Indices.any():
                #On supprime les matchs qui ont été reportés:
                Indices = [y for y in Indices if y <= (Indices[0] + ((constant_variables.nb_teams / 2) - 1))]
                
                #On calcule et on ajoute dans les colonnes du dataset, le rang avant le debut de la w eme journée de chaque team dans les 3 classements et on classe les potentielles equipes qui jouaient un match reporté juste avant le premier match de la journée w
                
                pnt_5lm_list=[]
                pnt_1lm_list=[]
                pnt_3lm_list=[]
                goal_diff_5lm_list=[]
                goal_diff_3lm_list=[]
                goal_diff_1lm_list=[]
                ranking_5lm_list=[]
                ranking_3lm_list=[]
                ranking_1lm_list=[]

                #On classe les équipes avant le début de la w ème journée:
                for y in Indices:
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_5lm_list, pnt_5lm_list , goal_diff_5lm_list, "home", 5, dico_col_rk_0, dataset_0)
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_5lm_list, pnt_5lm_list , goal_diff_5lm_list, "away", 5, dico_col_rk_0, dataset_0)
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_3lm_list, pnt_3lm_list , goal_diff_3lm_list, "home", 3, dico_col_rk_0, dataset_0)
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_3lm_list, pnt_3lm_list , goal_diff_3lm_list, "away", 3, dico_col_rk_0, dataset_0)
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_1lm_list, pnt_1lm_list , goal_diff_1lm_list, "home", 1, dico_col_rk_0, dataset_0)
                    useful_functions.classement_team_on_X_last_matchs(y, ranking_1lm_list, pnt_1lm_list , goal_diff_1lm_list, "away", 1, dico_col_rk_0, dataset_0)

                
                #On rajoute au classement les équipes manquantes (en cas de matchs reportés):

                useful_functions.ajout_missing_teams_ranking_on_X_last_matchs(y, ranking_5lm_list, pnt_5lm_list , goal_diff_5lm_list, 5, dico_col_rk_0, dataset_0)
                useful_functions.ajout_missing_teams_ranking_on_X_last_matchs(y, ranking_3lm_list, pnt_3lm_list , goal_diff_3lm_list, 3, dico_col_rk_0, dataset_0)
                useful_functions.ajout_missing_teams_ranking_on_X_last_matchs(y, ranking_1lm_list, pnt_1lm_list , goal_diff_1lm_list, 1, dico_col_rk_0, dataset_0)

                
                #On remplit pour chaque match de la w ème journée le dataset avec les classements prematch des HT et AT (uniquement si il s'agit d'une journée de championnat superieure au nombre de journées sur les quelles sont calculé le classement):
                if dataset_0.at[Indices[0],"Game Week"]>5:
                    useful_functions.fill_dataset_with_teams_rank_on_X_last_matchs(Indices, ranking_5lm_list, 5, dico_col_rk_0, dataset_0)
                if dataset_0.at[Indices[0],"Game Week"]>3:
                    useful_functions.fill_dataset_with_teams_rank_on_X_last_matchs(Indices, ranking_5lm_list, 3, dico_col_rk_0, dataset_0)
                if dataset_0.at[Indices[0],"Game Week"]>1:
                    useful_functions.fill_dataset_with_teams_rank_on_X_last_matchs(Indices, ranking_5lm_list, 1, dico_col_rk_0, dataset_0)
                
                #Si le ou les matchs qui précèdent (chronologiquement et donc aussi dans l'ordre des lignes du dataset) le premier de la journée w,
                #sont des matchs reportés, alors on va classer ces equipes dans les classements réalisés précedemment pour la journée w, en ayant 
                #préalablement retiré les equipes en question des classements
                
                useful_functions.classage_teams_playing_postponned_macth_on_X_last_matchs(Indices, w, start_date, end_date, ranking_5lm_list, pnt_5lm_list , goal_diff_5lm_list, 5, dico_col_rk_0, dataset_0)
                useful_functions.classage_teams_playing_postponned_macth_on_X_last_matchs(Indices, w, start_date, end_date, ranking_3lm_list, pnt_3lm_list , goal_diff_3lm_list, 3, dico_col_rk_0, dataset_0)
                useful_functions.classage_teams_playing_postponned_macth_on_X_last_matchs(Indices, w, start_date, end_date, ranking_1lm_list, pnt_1lm_list , goal_diff_1lm_list, 1, dico_col_rk_0, dataset_0) 
        
    return dataset_0
    #OK
    # Vérifié à tous niveaux mais compliqué car bcp de choses à check!

#CORNERS NB (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               X
#PER MATCH AVG HT/AT DIFF    V
def corners_nb(dico_col_rk_0, dataset_0):
    #On remplit les colonnes "HT_corners_nb", "AT_corners_nb" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTCN'], 20, 1, dataset_0)
          
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_corners_nb"] = (dataset_0["HT_corners_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_corners_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_corners_nb"] = (dataset_0["AT_corners_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_corners_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#YELLOW, RED CARDS NB (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               X
#PER MATCH AVG HT/AT DIFF    V
def yellow_red_cards(dico_col_rk_0, dataset_0):
    #On est obligés de garder l'ensemble du code pour ces deux variables car les colonnes ht_yeellow_cards, at_yellow_cards ne sont pas 
    #l'une à coté de l'autre. La fonction crée ne fonctionne pas dans ce cas la.

    #sert pour la boucle qui va permettre de remplir les dictionnaires "yellow_cards_nb_dico", "red_cards_nb_dico"
    nb_matches_trates=0
    rownb_last_season_match=0

    for i in (constant_variables.seasons):
        yellow_cards_nb_dico={}
        red_cards_nb_dico={}

        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison 
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)

        #On initialise le dico avec les noms des équipes
        yellow_cards_nb_dico = useful_functions.init_dico_with_names_teams(equipes)
        red_cards_nb_dico = useful_functions.init_dico_with_names_teams(equipes)
        
        rownb_last_season_match+=df.shape[0]
        
    #On remplit le dico qui contient le nom des 20 (ou 18) equipes et le nombre de yellow/red cards prematch des équipes.
    #On remplit également les 2 colonnes nb yellow/red cards pre-match de la Home et Away team.
        
        for j in range(nb_matches_trates, rownb_last_season_match):
            
    #On remplit les colonnes "HT_yellow_cards_nb", "AT_yellow_cards_nb", "HT_red_cards_nb", "AT_red_cards_nb" et on met à jour les dictionnaires
            dataset_0.iloc[j,dico_col_rk_0['rg_HTYCN']] = yellow_cards_nb_dico[dataset_0.iloc[j,4]]
            dataset_0.iloc[j,dico_col_rk_0['rg_ATYCN']] = yellow_cards_nb_dico[dataset_0.iloc[j,5]]
            dataset_0.iloc[j,dico_col_rk_0['rg_HTRCN']] = red_cards_nb_dico[dataset_0.iloc[j,4]]
            dataset_0.iloc[j,dico_col_rk_0['rg_ATRCN']] = red_cards_nb_dico[dataset_0.iloc[j,5]]
            
            yellow_cards_nb_dico[dataset_0.iloc[j,4]]+=dataset_0.iloc[j,22]
            yellow_cards_nb_dico[dataset_0.iloc[j,5]]+=dataset_0.iloc[j,24]
            red_cards_nb_dico[dataset_0.iloc[j,4]]+=dataset_0.iloc[j,23]
            red_cards_nb_dico[dataset_0.iloc[j,5]]+=dataset_0.iloc[j,25]

        nb_matches_trates+=df.shape[0]
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_red_cards_nb"] = (dataset_0["HT_red_cards_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_red_cards_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_red_cards_nb"] = (dataset_0["AT_red_cards_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_red_cards_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))

    dataset_0["HT_Diff_avg_yellow_cards_nb"] = (dataset_0["HT_yellow_cards_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_yellow_cards_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_yellow_cards_nb"] = (dataset_0["AT_yellow_cards_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_yellow_cards_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#SHOTS NB (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def shots_nb(dico_col_rk_0, dataset_0):
    #On remplit les colonnes "HT_shots_nb", "AT_shots_nb" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTSN'], 30, 1, dataset_0)

    #PER MATCH AVG
    dataset_0['HT_avg_shots_nb'] = dataset_0["HT_shots_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0['AT_avg_shots_nb'] = dataset_0["AT_shots_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_shots_nb"] = (dataset_0["HT_shots_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_shots_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_shots_nb"] = (dataset_0["AT_shots_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_shots_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#SHOTS ON TARGET NB (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def shots_on_target_nb(dico_col_rk_0, dataset_0):
    #On remplit les colonnes "HT_shots_on_target_nb", "AT_shots_on_target_nb" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTSOTN'], 32, 1,  dataset_0)
        
    #PER MATCH AVG
    dataset_0["HT_avg_shots_on_target_nb"] = dataset_0["HT_shots_on_target_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_shots_on_target_nb"] = dataset_0["AT_shots_on_target_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_shots_on_target_nb"] = (dataset_0["HT_shots_on_target_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_shots_on_target_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_shots_on_target_nb"] = (dataset_0["AT_shots_on_target_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_shots_on_target_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#FOULS NB (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def fouls_nb(dico_col_rk_0, dataset_0):
    #On remplit les colonnes "HT_fouls_nb", "AT_fouls_nb" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTFN'], 36, 1, dataset_0)
        
        
    #PER MATCH AVG
    dataset_0["HT_avg_fouls_nb"] = dataset_0["HT_fouls_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_fouls_nb"] = dataset_0["AT_fouls_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_fouls_nb"] = (dataset_0["HT_fouls_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_fouls_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_fouls_nb"] = (dataset_0["AT_fouls_nb"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_fouls_nb"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0

#POSSESSION (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def possession(dico_col_rk_0, dataset_0):
    #On remplit les colonnes "HT_posession", "AT_possession" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTP'], 38, 1, dataset_0)
        
        
    #PER MATCH AVG
    dataset_0["HT_avg_possession"] = dataset_0["HT_possession"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_possession"] = dataset_0["AT_possession"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_possession"] = (dataset_0["HT_possession"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_possession"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_possession"] = (dataset_0["AT_possession"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_possession"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0


#EXPECTED GOALS / XG (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def expected_goals(dico_col_rk_0, dataset_0):
    
    #We ensure that the columns are defined as float columns (because i had an error that raised saying that they are int64 col)
    dataset_0["HT_xg"] = dataset_0["HT_xg"].astype(float)
    dataset_0["AT_xg"] = dataset_0["AT_xg"].astype(float)
    
    #On remplit les colonnes "HT_xg", "AT_xg" 
    useful_functions.variable_sum_computing(dico_col_rk_0['rg_HTXG'], 42, 1, dataset_0)
        
        
    #PER MATCH AVG
    dataset_0["HT_avg_xg"] = dataset_0["HT_xg"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_xg"] = dataset_0["AT_xg"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_xg"] = (dataset_0["HT_xg"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_xg"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_xg"] = (dataset_0["AT_xg"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_xg"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0



# 1/VICTORY ODDS  (pm, sbos)
#VARIABLE                    V
#PER MATCH AVG               V
#PER MATCH AVG HT/AT DIFF    V
def odds_victory_proba(dico_col_rk_0, dataset_0):
    
    #We ensure that the columns are defined as float columns (because i had an error that raised saying that they are int64 col)
    dataset_0["HT_odds_victory_proba"] = dataset_0["HT_odds_victory_proba"].astype(float)
    dataset_0["AT_odds_victory_proba"] = dataset_0["AT_odds_victory_proba"].astype(float)
    
    #On remplit les colonnes "HT_odds_victory_proba", "AT_odds_victory_proba" 
    #On est obligés de faire cette étape à la main car on calcule la SOMME DES INVERSES des odds de chaque match, ce qui n'est pas faisable avec la fonction variable_sum_computing
    nb_matchs_trates=0
    rownb_last_season_match=0

    for i in (constant_variables.seasons):

        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison 
        equipes, df = useful_functions.noms_teams_season_and_df(i, dataset_0)

        #On initialise le dico avec les noms des équipes
        odds_vic_proba_nb_dico = useful_functions.init_dico_with_names_teams(equipes)

        rownb_last_season_match+=df.shape[0]

        #On remplit le dico qui contient le nom des 20 (ou 18) equipes et le nombre de xxx prematch des équipes.
        #On remplit également les colonnes HT_odds_victory_proba, AT_odds_victory_proba prematch

        for j in range(nb_matchs_trates,rownb_last_season_match):

            #On remplit les colonnes "HT_xxx_nb", "AT_xxx_nb" et on met a jour le dictionnaire
            dataset_0.iloc[j,dico_col_rk_0['rg_HTOVP']] = odds_vic_proba_nb_dico[dataset_0.iloc[j,4]]
            dataset_0.iloc[j,dico_col_rk_0['rg_HTOVP'] + 1] = odds_vic_proba_nb_dico[dataset_0.iloc[j,5]]

            odds_vic_proba_nb_dico[dataset_0.iloc[j,4]]+=(1/useful_functions.un_ou_x(dataset_0.iloc[j,56]))
            odds_vic_proba_nb_dico[dataset_0.iloc[j,5]]+=(1/useful_functions.un_ou_x(dataset_0.iloc[j,58]))


        nb_matchs_trates+=df.shape[0]

    #PER MATCH AVG
    dataset_0["HT_avg_odds_victory_proba"] = dataset_0["HT_odds_victory_proba"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))
    dataset_0["AT_avg_odds_victory_proba"] = dataset_0["AT_odds_victory_proba"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))
        
    #PER MATCH AVG HT/AT DIFF
    dataset_0["HT_Diff_avg_odds_victory_proba"] = (dataset_0["HT_odds_victory_proba"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["AT_odds_victory_proba"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    dataset_0["AT_Diff_avg_odds_victory_proba"] = (dataset_0["AT_odds_victory_proba"]/(dataset_0["AT_played_matchs_nb"].apply(useful_functions.un_ou_x))) - (dataset_0["HT_odds_victory_proba"]/(dataset_0["HT_played_matchs_nb"].apply(useful_functions.un_ou_x)))
    
    return dataset_0