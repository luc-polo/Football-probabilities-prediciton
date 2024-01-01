""" 
This module is made to create the new features we want to use for our model, or at least to test their relevancy.
"""

import numpy as np
import sys
from dateutil.relativedelta import relativedelta


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
        
        name_teams_season, d = useful_functions.noms_teams_season_and_df(i, dataset_0)
        
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
    budgets_2015={"PSG": 480, 'Toulouse' : 32, 'Olympique Marseille': 105, 'Rennes': 40, 'Nice': 40, 'Olympique Lyonnais':115, 'Bastia':22, 'Lille':65, 'Reims': 30, 'Lorient': 36, 'Caen':26, 'Nantes':32, 'Metz':28, 'Thonon Evian FC':28, 'Montpellier':40, 'Saint-Étienne':50, 'Bordeaux':55, 'Guingamp':25, 'Monaco':160, 'Lens':38 }
    budgets_2016={"PSG": 490, 'Toulouse' : 30, 'Olympique Marseille': 120, 'Rennes': 43, 'Nice': 40, 'Olympique Lyonnais':170, 'Bastia':25, 'Lille':75, 'Reims': 31, 'Lorient': 30, 'Caen':26, 'Nantes':38, 'Troyes':23, 'Angers SCO':24, 'Montpellier':42, 'Saint-Étienne':68, 'Bordeaux':55, 'Guingamp':25, 'Monaco':140, 'Gazélec Ajaccio':25 }
    budgets_2017={"PSG": 503, 'Toulouse' : 37, 'Olympique Marseille': 100, 'Rennes': 50, 'Nice': 45, 'Olympique Lyonnais':235, 'Nancy':30, 'Lille':75, 'Metz': 30, 'Lorient': 36, 'Caen':29, 'Nantes':40, 'Dijon':26, 'Angers SCO':29, 'Montpellier':42, 'Saint-Étienne':70, 'Bordeaux':60, 'Guingamp':26, 'Monaco':145, 'Bastia':28 }
    budgets_2018={"PSG": 540, 'Toulouse' : 34, 'Olympique Marseille': 120, 'Rennes': 50, 'Nice': 45, 'Olympique Lyonnais':240, 'Amiens SC':25, 'Lille':90, 'Metz': 33, 'Strasbourg': 30, 'Caen':32, 'Nantes':45, 'Dijon':32, 'Angers SCO':28, 'Montpellier':43.5, 'Saint-Étienne':68, 'Bordeaux':65, 'Guingamp':26, 'Monaco':170, 'Troyes': 26 }
    budgets_2019={"PSG": 500, "Olympique Lyonnais": 285, "Monaco": 215, "Olympique Marseille": 150, "Lille": 90, "Saint-Étienne": 74, "Bordeaux": 70, "Rennes": 68, "Nantes": 60, "Nice": 50, "Montpellier": 41.4, "Reims": 40, "Strasbourg": 37.5, "Amiens SC": 36, "Dijon": 35, "Toulouse": 35, "Caen": 34, "Angers SCO": 30, "Guingamp": 30, "Nîmes": 20} 
    budgets_2020={"PSG": 637, "Olympique Lyonnais": 310, "Monaco": 220, "Olympique Marseille": 110, "Lille": 120, "Saint-Étienne": 100, "Bordeaux": 70, "Rennes": 65, "Nantes": 70, "Nice": 50, "Montpellier": 40, "Reims": 45, "Strasbourg": 43, "Amiens SC": 30, "Dijon": 38, "Toulouse": 35, "Nîmes": 27, "Angers SCO": 32, "Brest": 30, "Metz": 40} 
    budgets_2021={"PSG": 640, "Olympique Lyonnais": 285, "Monaco": 215, "Olympique Marseille": 140, "Lille": 147, "Saint-Étienne": 95, "Bordeaux": 65, "Rennes": 105, "Nantes": 75, "Nice": 75, "Montpellier": 54.5, "Reims": 70, "Strasbourg": 50, "Lens": 46, "Dijon": 50, "Lorient": 45, "Nîmes": 40, "Angers SCO": 45, "Brest": 35, "Metz": 50} 
    budgets_2022={"PSG": 500, "Olympique Lyonnais": 250, "Monaco": 215, "Olympique Marseille": 250, "Lille": 147, "Saint-Étienne": 70, "Bordeaux": 112, "Rennes": 110, "Nantes": 65, "Nice": 90, "Montpellier": 44, "Reims": 60, "Strasbourg": 45, "Lens": 43, "Troyes": 30, "Lorient": 50, "Clermont": 20, "Angers SCO": 41, "Brest": 32, "Metz": 50}
    budgets_2023={"PSG": 700, "Olympique Lyonnais": 250, "Monaco": 240, "Olympique Marseille": 250, "Lille": 100, "Auxerre": 32, "Ajaccio": 22, "Rennes": 90, "Nantes": 75, "Nice": 100, "Montpellier": 52, "Reims": 70, "Strasbourg": 45, "Lens": 62, "Troyes": 45, "Lorient": 50, "Clermont": 25, "Angers SCO": 40, "Brest": 48, "Toulouse": 40}

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