""" 
This module is made to define useful functions we will use in several others functions or tasks.
For many functions below I have not completed the 'Args' and 'Returns' sections. I considered that the summury was suffiscient to understand the role and functioning of the function.
"""
#Import classic Python packages
from dateutil.relativedelta import relativedelta
import pandas as pd

#Import our modules
if __name__ == '__main__':
    from data.constant_variables import *

def un_ou_x(x):
    """ 
        Function that returns 1 if x=0, else x
        
    Args:
    
    Returns:
    
    """
    if x == 0: 
        return 1
    else:
        return x

def noms_teams_season_and_df(i, dataset_0):
    """ 
        Function that returns a list with the name of each team playing in the championship for a specific season, as well as a dataframe containing only the matchs/lines of this particular season.
    
    Args:
        i (datetime): The end date of the season we want to get the matches and the teams from
        
        dataset_0 (DataFrame): The dataframe in which we got the matches of the season (dataset in our situation)
    
    Returns:
        tuple:
            A tuple containing:
            
            * **list**: A list containing the names of the teams that have competed during this season in the championship.
              So there are 18 or 20 values in this list (I only know championships with 20 or 18 teams).

            * **DataFrame**: DataFrame containing only the matches/lines of this particular season (defined by its end date 'i').
    """
    start_date = i - relativedelta(years=1)
    end_date = i
    df=dataset_0.loc[(start_date < dataset_0["date_GMT"]) & (dataset_0["date_GMT"] <= end_date),["home_team_name"]]
    return df["home_team_name"].unique(), df

def init_dico_with_names_teams(team_names):
    """ 
        Function made to initialise a dictionnary with teams names as key and 0 as value. It is complementary to noms_teams_season_and_df(). For each season we use the dictionnaries generated by these functions to stock teams statistics when we calculate them. 
    
    Args:
        team_names (list): List that contains the names of the teams we will put in dictionnary. 
    
    Returns:
        dictionnary 
    """
    
    dico={}
    for e in team_names:
        dico.update({e:0})
    return dico

def find_rank(ranking_list_0, pnt_list_0, gd_list_0, current_points_0, current_goal_diff_0):
    """ 
        Find the rank (position) for a team in a ranking, (the ranking are materialised by list where the order of teams names represent their rank) based on points and goal difference.
    
    Args:
    
    Returns:
        int: The computed rank for the team.
    """
    rank_0 = 0
    while rank_0 < len(ranking_list_0) and (pnt_list_0[rank_0] > current_points_0 or (pnt_list_0[rank_0] == current_points_0 and gd_list_0[rank_0] > current_goal_diff_0)):
        rank_0 += 1
    return rank_0

def insert_func(team_0, ranking_list_0, pnt_list_0, gd_list_0, team_points_0, team_gd_0, team_rank_0):
    """ 
        Insert the name of a team and its 2 ranking statistics into the ranking lists dedicated, given the rank (determined before by find_rank()) of this team in the ranking. Every stat, as the name of the team, is positionned in its list at the position of the team's rank. For instance if team_rank_0 = 5, team_0, team_points_0 and team_gd_0 will be inserted at the 5th place in the dedicated lists.
        
    Args:
        team_0 (str): The name of the team we want to insert the stats and the name in the lists.
        
        ranking_list_0 (list): List that serves as the ranking. It contains the names of the teams. The position of a name in the list represents its rank.
        
        pnt_list_O (list): List containing the teams points numbers.
        
        gd_list_0 (list): List containing the teams goal difference numbers.
        
        team_points_0 (int): Points number of team_0, to insert in pnt_list_O.
        
        team_gd_0 (int): Goal difference of team_0, to insert in gd_list_0.
        
        team_rank_0 (int): Rank of team_0. That's the position at which we will insert the team's statistics and name in the lists.
    
    Returns:
    
    """
    ranking_list_0.insert(team_rank_0,team_0)
    pnt_list_0.insert(team_rank_0, team_points_0)
    gd_list_0.insert(team_rank_0, team_gd_0)

def classement_team(y_0, ranking_0, pnt_list_0 , goal_diff_list_0, H_A_T_0, dico_col_rk, dataset_0):
    """ 
            Given a line number and a Home or Away status, this function ranks a team (among the other teams, prematch) based on its statistics. Then it positions the team stats and its name at the appropriate rank in ranking, pnt_list , goal_diff_list. 
        
        Args:
            y_0 (int): Line number in the dataset of the team match we want to compute the prematch rank. 
            
            ranking_0 (list): Ranking list in which we will position team's name
            
            pnt_list (list): List of the teams point in which we will insert, at the correct rank, the team points number.
            
            goal_diff_list_0 (list): List of the teams goal differences in which we will insert, at the correct rank, the team goal difference.
            
            H_A_T_0 (str): 'HT' or 'AT' to inform wether the team we want to compute the ranking is the HT or AT on this match.
            
            dataset_0 (DataFrame): DataFrame containing our data.
        
        Returns:
            None
    """
    
    #On définit les numéros des colonnes qu'on va utiliser en fonction de si on traite la team qui joue à dom ou à l'ext
    if H_A_T_0 == "home":
        rg_PPN=dico_col_rk['rg_PHTPN']
        rg_gdPM =  dico_col_rk['rg_gdHTPM']
        name_team = dataset_0.iloc[y_0,4]
    else:
        rg_PPN= dico_col_rk['rg_PATPN']
        rg_gdPM =  dico_col_rk['rg_gdATPM']
        name_team = dataset_0.iloc[y_0,5]
                       
    k=dataset_0.iloc[y_0, dico_col_rk['rg_PPN']]
    #gd= goal diff de la team qu'on classe
    gd=dataset_0.iloc[y_0, dico_col_rk['rg_gdPM']]
    #On classe la team de la y_0 ème ligne pour la w ème journée:
    rank = find_rank(ranking_0, pnt_list_0, goal_diff_list_0, k, gd)
    
    #On place l'équipe considérée à son rang précédemment calculé dans nos trois listes     
    insert_func(name_team, ranking_0, pnt_list_0, goal_diff_list_0, k, gd, rank)

def classement_team_on_X_last_matchs(y_0, ranking_0, pnt_list_0 , goal_diff_list_0, H_A_T_0, x_last_matches, dataset_0):
    """ 
        Given a line number and a Home or Away status, this function computes the rank of a team based on its performances/stats on the x_last_matches. Then it positions the team stats and its name at the appropriate rank in ranking_0, pnt_list_0, goal_diff_list_0.
        It makes the same thing as classement_team(), but with one difference: It uses the statistics on the X last matches (which are not located in the same dataset columns), to compute the rank. 
        
    Args:
        y_0 (int): Line number in the dataset of the team match we want to compute the prematch rank on x_last_matches. 
        
        ranking_0 (list): Ranking list in which we will position team's name. That's the ranking on the x_last_matches.
        
        pnt_list_0 (list): List of the teams points on the x_last_matches, in which we will insert, at the correct rank, the team points number on the x_last_matches.
        
        goal_diff_list_0 (list): List of the teams goal differences on the x_last_matches, in which we will insert, at the correct rank, the team goal difference on the x_last_matches.
        
        H_A_T_0 (str): 'HT' or 'AT' to inform wether the team we want to compute the ranking on x_last_matches is the HT or AT on this match.
        
        x_last_matches (int): The number of most recent matches considered to compute teams statistics and set the rankings. x_last_matches is included in [1, 3, 5].
        
        dataset_0 (DataFrame): DataFrame containing our data.
    
    Returns:
        None
    
    """
    
    #on definit le décalage de chaque colonne Points_HT_Xlm_PM et Points_AT_Xlm_PM par rapport à Points_HT_1lm_PM en fonction de x_last_matchs
    decalage = 0
    
    if x_last_matches == 3: 
        decalage = 2
    if x_last_matches == 1: 
        decalage = 4
    
    #On définit les numéros des colonnes qu'on va utiliser en fonction de si on traite la team qui joue à dom ou à l'ext
    if H_A_T_0 == "home":
        rg_PPN = rg_PHT5LMPM + decalage
        rg_gdPM =  rg_GDHT5LMPM + decalage
        name_team = dataset_0.iloc[y_0,4]
    else:
        rg_PPN = rg_PAT5LMPM  + decalage
        rg_gdPM =  rg_GDAT5LMPM + decalage
        name_team = dataset_0.iloc[y_0,5]  
    
    #k = points number of the team we rank
    k=dataset_0.iloc[y_0,rg_PPN]
    #gd= goal diff of the team we rank
    gd=dataset_0.iloc[y_0, rg_gdPM]
    #On classe la team de la y_0 ème ligne pour la w ème journée:
    rank = find_rank(ranking_0, pnt_list_0, goal_diff_list_0, k, gd)
    
    #On place l'équipe considérée à son rang précédemment calculé dans nos trois listes     
    insert_func(name_team, ranking_0, pnt_list_0, goal_diff_list_0, k, gd, rank)

def ajout_missing_teams_ranking(y_0, ranking_0, pnt_list_0, goal_diff_list_0, dataset_0):
    """ 
        Add the teams missing in a Game Week ranking (because of postponed matches). The function, starting from one line of the Game Week concerned, iterates through the dataset by descending line numbers, to find the last matchs of teams missing, to add the teams in the ranking.
    
    Args:
        y_0 (int): Line number in the dataset we will start from to search the last matches of the teams missing.
        
        ranking_0 (list): Ranking list we want to complete with missing teams names.
        
        pnt_list_0 (list): Points list we want to complete with missing teams values.
        
        goal_diff_list_0 (list): Goal diff list we want to complete with missing teams values.
        
        dataset_0 (DataFrame): DataFrame containing our data.
        
    Returns:
        None
    """
    while len(ranking_0)< nb_teams:
        while dataset_0.iloc[y_0,4] in ranking_0 and dataset_0.iloc[y_0,5] in ranking_0:
            y_0+=1
        if dataset_0.iloc[y_0,4] not in ranking_0:
            classement_team(y_0, ranking_0, pnt_list_0 , goal_diff_list_0, "home")

        if dataset_0.iloc[y_0,5] not in ranking_0:
            classement_team(y_0, ranking_0, pnt_list_0 , goal_diff_list_0, "away")

#Fonction qui permet de rajouter les equipes manquantes (pour cause de match reporté) aux classements sur les X derniers matchs:
def ajout_missing_teams_ranking_on_X_last_matchs(y_0, ranking_0, pnt_list_0, goal_diff_list_0, x_last_matches, dataset_0):
    """ 
        Add the teams missing in a Game Week ranking on the X last matches (because of postponed matches). The function, starting from one line of the Game Week concerned, iterates through the dataset by descending line numbers, to find the last matchs of teams missing, to add the them in the ranking on the X last matches.
        It makes the same thing as ajout_missing_teams_ranking(), but with one difference: It uses the statistics on the X last matches (which are not located in the same dataset columns), to compute the rank. 
    
    Args:
        y_0 (int): Line number in the dataset we will start from to search the last matches of the teams missing.
        
        ranking_0 (list): Ranking list on x_last_matches, we want to complete with missing teams names.
        
        pnt_list_0 (list): Points list on x_last_matches, we want to complete with missing teams values.
        
        goal_diff_list_0 (list): Goal diff list on x_last_matches, we want to complete with missing teams values.
        
        x_last_matches (int): The number of most recent matches considered to compute teams statistics and set the rankings. x_last_matches is included in [1, 3, 5].
                
        dataset_0 (DataFrame): DataFrame containing our data.
        
    Returns:
        None
    """
    y_local = y_0
    while len(ranking_0)< nb_teams:
        while dataset_0.iloc[y_local,4] in ranking_0 and dataset_0.iloc[y_local,5] in ranking_0:
            y_local+=1
        if dataset_0.iloc[y_local,4] not in ranking_0:
            classement_team_on_X_last_matchs(y_local, ranking_0, pnt_list_0 , goal_diff_list_0, "home", x_last_matches)

        if dataset_0.iloc[y_local,5] not in ranking_0:
            classement_team_on_X_last_matchs(y_local, ranking_0, pnt_list_0 , goal_diff_list_0, "away", x_last_matches)

#Remplit pour chaque match de la w ème journée le dataset avec les classements prematch des HT et AT:
def fill_dataset_with_teams_rank(Indices_0, ranking_0, dataset_0):
    """ 
        Fill, for each match of the w th Game Week, the dataset with the prematch ranks of teams (input with ranking_0). The prematch ranks are input into the function through a ranking list (containing teams names oredered following their rank).
    
    Args:
        Indices_0 (list): List of indices corresponding to the Game Week match lines in the dataset_0. These are the indices of the lines where the function will fill in the prematch ranks of the teams.
        
        ranking_0 (list): Ranking list that contains teams names oredered following their rank. This list is utilized to stock the prematch ranks of teams, which are then filled into the dataset.
        
        dataset_0 (DataFrame): DataFrame containing our data.
    
    Returns:
        None
    """
    
    for y in Indices_0:  
                dataset_0.iloc[y, rg_HTWR]= ranking_0.index(dataset_0.iloc[y,4]) + 1
                dataset_0.iloc[y, rg_ATWR]= ranking_0.index(dataset_0.iloc[y,5]) + 1 

#Remplit pour chaque match de la w ème journée le dataset avec les classements prematch sur les X last matchs des HT et AT:   
def fill_dataset_with_teams_rank_on_X_last_matchs(Indices_0, ranking_0, x_last_matches, dataset_0):
    """ 
        Exactly the same as fill_dataset_with_team_rank() but for the rankings in x last matchs
    
    Args:
        Indices_0 (list): List of indices corresponding to the Game Week match lines in the dataset_0. These are the indices of the lines where the function will fill in the prematch ranks on the x last matches of the teams.
        
        ranking_0 (list): Ranking list that contains teams names oredered following their rank on the x_last_matches. This list is utilized to stock the prematch ranks of teams on x last matches, which are then filled into the dataset.
        
        x_last_matches (int): The number of most recent matches considered to compute teams statistics and set the rankings. x_last_matches is included in [1, 3, 5].
        
        dataset_0 (DataFrame): DataFrame containing our data.
    
    Returns:
        None
    """
    decalage = 0
    if x_last_matches == 3: 
        decalage = 2
    if x_last_matches == 1: 
        decalage = 4
    
    for y in Indices_0:  
                dataset_0.iloc[y, rg_HT5lm_WR + decalage]= ranking_0.index(dataset_0.iloc[y,4]) + 1
                dataset_0.iloc[y, rg_AT5lm_WR + decalage]= ranking_0.index(dataset_0.iloc[y,5]) + 1

#Permet de supprimer les equipes du match reporté en question, des classements de la w eme journee
def delete_postponned_match_teams_from_ranking(ranking_0, pnt_list_0, goal_diff_list_0, l_0, dataset_0):
    """ 
        Delete the teams of a given match in the ranking and stats lists of the Game Week concerned. (We use it to delete postponed matchs from the rankings).  It's a function used in classage_teams_playing_postponned_macth_on_X_last_matchs(). See this function description to understand the use of this function.
    
    Args:
    
    Returns:
        None 
    """
    rk_ht = ranking_0.index(dataset_0.iloc[l_0,4])
    rk_at = ranking_0.index(dataset_0.iloc[l_0,5])

    rk_t_sorted= sorted([rk_ht, rk_at], reverse = True)
                    
    for rk1 in rk_t_sorted:                        
        pnt_list_0.pop(rk1)
        ranking_0.pop(rk1)
        goal_diff_list_0.pop(rk1)

#Si le ou les matchs qui précèdent (chronologiquement et donc aussi dans l'ordre des lignes du dataset) le premier de la journée w, sont des matchs reportés, alors on va classer ces equipes dans le classement réalisé précedemment pour la journée w, en ayant préalablement retiré les equipes en question du classement
def classage_teams_playing_postponned_macth_on_X_last_matchs(Indices_0, w_0, start_date_0, end_date_0, ranking_0, pnt_list_0, goal_diff_list_0, x_last_matches_0, dataset_0):
    """ 
        When the match(s) that precede (chronologically and therfore in the order of dataset lines) the first match of the Game Week w_0, are matches postponned, we rank, on the x last matches, the teams of these postponed matchs in the ranking previously made for the Game Week w. Before ranking the teams in this ranking, we firstly remove these teams from the ranking on x last matchs with delete_postponned_match_teams_from_ranking().
        That's what this function makes.
    
    Args:
        Indices_0 (list): List of indices corresponding to the lines of the w th Game Week matchs in the dataset_0.
        
        w_0 (int): The Game Week before which the function will check if there are one or more postponed matches.
        
        start_date_0 (datetime): Start date of the season we are working on. It's made to avoid the algo considers the match of the precedent season.
        
        end_date_0 (datetime): End date of the season we are working on. It's made to avoid the algo considers matchs of next season.
        
        ranking_0 (list): The ranking on the x_last_matches of the w_0 Game Week established so far and in which we will rank the teams which played postponned match before w_0 (if there are some).
        
        pnt_list_0 (list): Points list on the x_last_matches of the w_0 Game Week established so far and which we will use to rank the teams which played postponned match before w_0 (if there are some).
        
        goal_diff_list_0 (list): Goal diff list on the x_last_matches of the w_0 Game Week established so far and which we will use to rank the teams which played postponned match before w_0 (if there are some).
        
        x_last_matches (int): The number of most recent matches considered to compute teams statistics and set the rankings. x_last_matches is included in [1, 3, 5].
        
        dataset_0 (DataFrame): DataFrame containing our data.
        
    Returns:
        None
    """
    #on definit le décalage de chaque colonne ..._Xlm_PM par rapport ..._5lm_PM en fonction de x_last_matchs
    decalage = 0
    if x_last_matches_0 == 3: 
        decalage = 2
    if x_last_matches_0 == 1: 
        decalage = 4
    
    nb_line_last_match_of_week = min(Indices_0)
    
    if nb_line_last_match_of_week != 0 :
        l= nb_line_last_match_of_week - 1

        
        #La condition qui suit le while permet de vérifier que la ligne l correspond à un match repporté
        #La seconde évite de considérer des matchs des saisons d'avant ou d'après
        #La troisième vérifie que le match reporté n'ait pas déja un classement rempli dans la case en question.
        while dataset_0.at[l, "Game Week"] < (w_0-1) and (start_date_0 < dataset_0.at[l,"date_GMT"]) and (dataset_0.at[l,"date_GMT"] <= end_date_0) and (dataset_0.iloc[l,rg_HT5lm_WR + decalage] == 0):            
            #On supprime les equipes du match reporté en question, des classements de la w eme journee
            delete_postponned_match_teams_from_ranking(ranking_0, pnt_list_0, goal_diff_list_0, l)

            #On classe les equipes du match reporté dans les classements de la w eme journee
            classement_team_on_X_last_matchs(l, ranking_0, pnt_list_0, goal_diff_list_0, "home", x_last_matchs_0)
            classement_team_on_X_last_matchs(l, ranking_0, pnt_list_0, goal_diff_list_0, "away", x_last_matchs_0)
            
            #On remplit les colonnes du dataset avec les classements des 2 equipes:            
            dataset_0.iloc[l, rg_HT5lm_WR + decalage]= ranking_0.index(dataset_0.iloc[l,4]) + 1
            dataset_0.iloc[l, rg_AT5lm_WR + decalage]= ranking_0.index(dataset_0.iloc[l,5]) + 1
            l-=1

#Calculate Variable sum since the beginning of the season. It's then used for Per Match Avg, Per Match Avg HT/AT Diff computing.
#We should change the name by 'variable_sum_computing' or 'calculation_variable_sum'
def calculation_variable_avg_diff(rg_ht_variable_nb, rg_ht_variable_ds_fichier_csv, space_between_ht_at_variables, dataset_0):
    """ 
        For one season and all the teams, the function calculates, for each match, the prematch sum of a given variable since the beginning of the season. The function also fills the column reserved for this purpose with the sum of this particular variable. The sums of variables are then used to compute the 'per match avg' and 'per match avg HT/AT Diff' (but that's not included in this function).
    
    Args:
        rg_ht_variable_nb (int): Rank of the column (in the dataframe input) to fill with the sum of the variable for HT (home team). 
        
        rg_ht_variable_ds_fichier_csv (int): Rank of the column (in the dataframe input) that contains the variable values per match, for HT. The variable values per match are the values the function sums.
        
        space_between_ht_at_variables (int): Number of columns between the HT and AT columns for the variable. Some variables as 'yellow cards nb' and 'red cards nb' have not the columns for HT and AT side by side. As we deduce the rank of AT columns from HT columns rank input (rg_ht_variable_nb, rg_ht_variable_ds_fichier_csv), we need to know how many columns are located between HT and AT columns. Most of the time it's 0, but sometimes it's 1 or 2.
    
        dataset_0 (DataFrame): DataFrame containing our data.
        
    Returns:
        None
    """
    nb_matchs_trates=0
    rownb_last_season_match=0

    for i in (saisons):
        variable_nb_dico={}

        #On créé "equipes" qui contient les noms de toutes les équipes du championnat durant une saison 
        equipes, df = noms_teams_season_and_df(i, dataset_0)

        #On initialise le dico avec les noms des équipes
        variable_nb_dico = init_dico_with_names_teams(equipes)

        rownb_last_season_match+=df.shape[0]

    #On remplit le dico qui contient le nom des 20 (ou 18) equipes et le nombre de xxx prematch des équipes.
    #On remplit également les colonnes nb xxx prematch de la Home et Away team.

        for j in range(nb_matchs_trates,rownb_last_season_match):

    #On remplit les colonnes "HT_xxx_nb", "AT_xxx_nb" et on met a jour le dictionnaire
            dataset_0.iloc[j,rg_ht_variable_nb] = variable_nb_dico[dataset_0.iloc[j,4]]
            dataset_0.iloc[j,rg_ht_variable_nb + 1] = variable_nb_dico[dataset_0.iloc[j,5]]

            variable_nb_dico[dataset_0.iloc[j,4]]+=dataset_0.iloc[j,rg_ht_variable_ds_fichier_csv]
            variable_nb_dico[dataset_0.iloc[j,5]]+=dataset_0.iloc[j,rg_ht_variable_ds_fichier_csv + space_between_ht_at_variables]


        nb_matchs_trates+=df.shape[0]

#To merge HT et AT columns for a given subsets of features applying a selection with a min value for number of matchs played by every team
#We should rename this function 'HT_AT_col_concat'
def HT_AT_col_merger(names_col_to_concat, names_col_concat, min_value_nb_matchs_played, dataset_0):
    """ 
        Create a dataframe made of columns that are the concatenation of HT (home team) and AT (away team) columns of a given subset of variables, selecting only matches where Game Week > min_value_nb_matchs_played.
    
    Args:
        names_col_to_concat (list): List of the HT columns names we want to concat with AT columns.
        
        names_col_concat (list): List of the names we want to give to the concatenated columns, in the same order as names_col_to_concat.
        
        min_value_nb_matchs_played (int): Minimimum value (excluded) of played matchs number we will apply to select the matchs we include in the concatenations.
        
    Returns:
        DataFrame: The dataframe containing the concatenation of the HT and AT columns of the variables, selecting only matches where Game Week > min_value_nb_matchs_played.
    """
    Df_HT_AT_features_col_concatenated = pd.DataFrame()
    for i in range(len(names_col_concat)):
        col1 = dataset_0[dataset_0['HT_played_matchs_nb']>min_value_nb_matchs_played][names_col_to_concat[2*i]]
        col2 = dataset_0[dataset_0['AT_played_matchs_nb']>min_value_nb_matchs_played][names_col_to_concat[2*i+1]]
        concatenated_features_column = pd.concat([col1, col2], axis=0)
        Df_HT_AT_features_col_concatenated = pd.concat([Df_HT_AT_features_col_concatenated, concatenated_features_column], axis = 1)
    #On attribue aux col du dataframe concaténné les noms qu'on leur a défini dans la liste names_col_concat
    Df_HT_AT_features_col_concatenated.columns = names_col_concat  
    
    return Df_HT_AT_features_col_concatenated

def compare_2_df_excepted_col(col_not_to_compare,df1, df2):
    """  
    This function is used to compare two dataframes, saying if they are exactly the same, excluding one column from the comparaison. That's useful during the process of columns filling. For instance, when we've just filled a column with values, we want to make sure that other columns were not modified too. So we use this funct° to compare the dataframe before filling and the one after filling, excluding the column filled from the comparaison.
    
    Args:
        col_not_to_compare (list): Name(s) of the column(s) not to include in the comparaison
        
        df1 (DataFrame): The first dataframe to compare
        
        df2 (DataFrame): The second dataframe to compare
    
    Returns:
        boolean: Wether the both dataframe are the same at the exception of the col_not_to_compare.
    """
    df1_0 = df1.copy()
    df2_0 = df2.copy()
    
    # Remove col_not_to_compare from our datasets
    df1_0 = df1_0.drop(columns=col_not_to_compare)
    df2_0 = df2_0.drop(columns=col_not_to_compare)
    
    # Check if the DataFrames are equal except for the specified columns
    are_equal_except_column = df1_0.equals(df2_0)
    
    if are_equal_except_column:
        print(f"The DataFrames are equal except for columns {col_not_to_compare}")
        return True
    else:
        print("The DataFrames are not equal")
        return False