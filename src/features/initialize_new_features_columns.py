
""" 
This module is made to create new columns in the dataframe, dedicated to the new features created in make_new_features.py. We create the columns, name it, create variables that contain the columns rank.
"""

#Import classic python packages
import pandas as pd
import sys

# modify the sys.path list to include the path to the data directory that contains the modules that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data import constant_variables



def add_columns_and_complete_col_ranks(dataset_0):
    """  
        Create new columns in the dataframe, dedicated to the new features created in make_new_features.py. We create the columns, name it and assign their values to columns ranks variables we create. The columns ranks variables are returned in a dictionnary.
        The new columns are created following the existing ones, meaning on the right side of the dataframe.
        
    Args:
        dataset_0 (DataFrame): DataFrame containing our data.
        
    Returns:
        dataset_0 (DataFrame): Dataframe with columns added
    
        dico_col_ranks (Dictionnary): Dictionnarry containing the new columns ranks names as keys and their values as values.
        
    """
    #Initialize the dictionnary
    dico_col_ranks = {}
    L,l = dataset_0.shape
    
    #used to fill the new columns with 0 values
    new_col = [0 for i in range(L)]
    
                                    #STATISTIQUES DE RESULTAT


                #Colonnes RESULTAT HT et AT au terme du match de la ligne (after match)
    dataset_0["RH"] = new_col
    rg_RH= dataset_0.shape[1]-1
    dataset_0["RA"] = new_col
    rg_RA= dataset_0.shape[1]-1
    
    dico_col_ranks.update({'rg_RH': rg_RH, 'rg_RA': rg_RA})
                
        
                #Colonnes status Home or Away of teams
    dataset_0["HT_H_A_status"] = new_col
    rg_H_HAS = dataset_0.shape[1]-1 
    dataset_0["AT_H_A_status"] = new_col
    rg_A_HAS = dataset_0.shape[1]-1
    
    dico_col_ranks.update({'rg_H_HAS': rg_H_HAS, 'rg_A_HAS': rg_A_HAS})


                #MATCHS PLAYED NB and VICTORIES NB (pre match) (since the beginning of the season)
    columns = ['HT_played_matchs_nb', 'AT_played_matchs_nb', 'HT_victories_nb', 'AT_victories_nb']
    #The size of temp_df is determined by the number of rows in dataset_0 (since its index is set to be the same)
    #temp_df is initialized with zeros (0) for all values
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTPMN, rg_ATPMN, rg_HTVN, rg_ATVN = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTPMN':rg_HTPMN, 'rg_ATPMN':rg_ATPMN, 'rg_HTVN':rg_HTVN, 'rg_ATVN':rg_ATVN})


        #VICTORY NUMBER (pre match) (since the beginning of the season)
    #Per Match Avg
    #Per match Avg HT/AT Diff 
    columns = ['HT_avg_victory_pm', 'AT_avg_victory_pm', 'Diff_HT_avg_victory_pm', 'Diff_AT_avg_victory_pm']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTavgVpm, rg_ATavgVpm, rg_DHTAVPM, rg_DATAVPM = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTavgVpm':rg_HTavgVpm, 'rg_ATavgVpm':rg_ATavgVpm, 'rg_DHTAVPM':rg_DHTAVPM, 'rg_DATAVPM':rg_DATAVPM})


        #(POINTS COLLECTED x (1+ADVERSARY AVG VICTORY PER MATCH))  (pre match) (since the beginning of the season)
    #Variable
    #Per Match Avg
    #HT/AT Diff
    columns = ['HT_points_ponderated_by_adversary_perf', 'AT_points_ponderated_by_adversary_perf',
            'HT_avg_pm_points_ponderated_by_adversary_perf', 'AT_avg_pm_points_ponderated_by_adversary_perf',
            'HT_Diff_points_ponderated_by_adversary_perf', 'AT_Diff_points_ponderated_by_adversary_perf']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTPPBAP, rg_ATPPBAP, rg_HTAPPBAP, rg_ATAPPBAP, rg_HTDPPBAP, rg_ATDPPBAP = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3,ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTPPBAP':rg_HTPPBAP, 'rg_ATPPBAP':rg_ATPPBAP, 'rg_HTAPPBAP':rg_HTAPPBAP, 'rg_ATAPPBAP':rg_ATAPPBAP, 'rg_HTDPPBAP':rg_HTDPPBAP, 'rg_ATDPPBAP':rg_ATDPPBAP })


                #SCORED GOALS CONCEDED GOALS (pre match) (since the beginning of the season)
    columns = ['scored_goals_HT_PM', 'conceeded_goals_HT_PM', 'scored_goals_AT_PM', 'conceeded_goals_AT_PM']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_SGHTPM, rg_CGHTPM, rg_SGATPM, rg_CGATPM = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_SGHTPM':rg_SGHTPM, 'rg_CGHTPM':rg_CGHTPM, 'rg_SGATPM':rg_SGATPM, 'rg_CGATPM':rg_CGATPM })
        
        
        
        #GOAL DIFF (pre match) (since the beginning of the season)
    #Variable
    #Per Match Avg 
    #Per match Avg HT/AT Diff
    columns =['goal_diff_HT_PM', 'goal_diff_AT_PM', 'HT_avg_goal_diff_pm', 'AT_avg_goal_diff_pm','Diff_HT_goal_diff_pm',
            'Diff_AT_goal_diff_pm']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_gdHTPM, rg_gdATPM, rg_HTAGDPM, rg_ATAGDPM, rg_DHTGDPM, rg_DATGDPM = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3,ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_gdHTPM':rg_gdHTPM, 'rg_gdATPM':rg_gdATPM, 'rg_HTAGDPM':rg_HTAGDPM, 'rg_ATAGDPM':rg_ATAGDPM, 'rg_DHTGDPM':rg_DHTGDPM, 'rg_DATGDPM':rg_DATGDPM })
    
    
        #SCORED GOALS/CONCEEDED GOALS  (pre match) (since the beginning of the season)   
    #Variable 
    #Per match HT/AT Diff        
    columns =['HT_avg_scored_g_conceedded_g_ratio', 'AT_avg_scored_g_conceedded_g_ratio',
            'Diff_HT_avg_scored_g_conceedded_g_ratio', 'Diff_AT_avg_scored_g_conceedded_g_ratio']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]         
    rg_HTavgSGCGR, rg_ATavgSGCGR, rg_DHTASGCGR, rg_DATASGCGR = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTavgSGCGR':rg_HTavgSGCGR, 'rg_ATavgSGCGR':rg_ATavgSGCGR, 'rg_DHTASGCGR':rg_DHTASGCGR, 'rg_DATASGCGR':rg_DATASGCGR })


            #POINTS NUMBER (pre match) (since the beginning of the season)
    #Variable
    #Per Match Avg
    #Per match Avg HT/AT Diff
    columns = ['Prematch_HT_PN', 'Prematch_AT_PN', 'HT_avg_collected_points_pm', 'AT_avg_collected_points_pm',
            'Diff_pnt_HT_AT_ratio', 'Diff_pnt_AT_HT_ratio']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1] 
    rg_PHTPN, rg_PATPN, rg_HTavgCPpm, rg_ATavgCPpm, rg_Diff_pnt_HT_AT_ratio, rg_Diff_pnt_AT_HT_ratio  = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_PHTPN':rg_PHTPN, 'rg_PATPN':rg_PATPN, 'rg_HTavgCPpm':rg_HTavgCPpm, 'rg_ATavgCPpm':rg_ATavgCPpm, 'rg_Diff_pnt_HT_AT_ratio':rg_Diff_pnt_HT_AT_ratio, 'rg_Diff_pnt_AT_HT_ratio':rg_Diff_pnt_AT_HT_ratio })



            #RANKING (pre match) (since the beginning of the season)
    #Variable
    #HT/AT Diff
    columns = ['HT_week_ranking', 'AT_week_ranking', 'Diff_HT_ranking', 'Diff_AT_ranking']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTWR, rg_ATWR, rg_DHTR, rg_DATR = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTWR':rg_HTWR, 'rg_ATWR':rg_ATWR, 'rg_DHTR':rg_DHTR, 'rg_DATR':rg_DATR })


            #ANNUAL BUDGET (pre match) (since the beginning of the season)
    #Variable
    #HT/AT Diff
    columns = ['annual budget of HT', 'annual budget of AT', 'Diff_HT_annual_budget', 'Diff_AT_annual_budget']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_ABHT, rg_ABAT, rg_DHTAB, rg_DATAB = [ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_ABHT':rg_ABHT, 'rg_ABAT':rg_ABAT, 'rg_DHTAB':rg_DHTAB, 'rg_DATAB':rg_DATAB })


    #42


        #POINTS NB ON 1,3,5 LAST MATCHS 
    #Variable
    #HT/AT Diff
    columns = ['Points_HT_5lm_PM', 'Points_AT_5lm_PM', 'Points_HT_3lm_PM', 'Points_AT_3lm_PM', 'Points_HT_1lm_PM', 
            'Points_AT_1lm_PM', 'HT_Diff_Points_5lm', 'AT_Diff_Points_5lm', 'HT_Diff_Points_3lm', 'AT_Diff_Points_3lm',
            'HT_Diff_Points_1lm', 'AT_Diff_Points_1lm']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_PHT5LMPM, rg_PAT5LMPM, rg_PHT3LMPM, rg_PAT3LMPM, rg_PHT1LMPM, rg_PAT1LMPM, rg_HTDP5LM, rg_ATDP5LM, rg_HTDP3LM, rg_ATDP3LM, rg_HTDP1LM, rg_ATDP1LM = [ds_len - 12, ds_len - 11, ds_len - 10, ds_len - 9, ds_len - 8, ds_len - 7, ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_PHT5LMPM':rg_PHT5LMPM, 'rg_PAT5LMPM':rg_PAT5LMPM, 'rg_PHT3LMPM':rg_PHT3LMPM, 'rg_PAT3LMPM':rg_PAT3LMPM, 'rg_PHT1LMPM':rg_PHT1LMPM, 'rg_PAT1LMPM':rg_PAT1LMPM, 'rg_HTDP5LM':rg_HTDP5LM, 'rg_ATDP5LM':rg_ATDP5LM, 'rg_HTDP3LM':rg_HTDP3LM, 'rg_ATDP3LM':rg_ATDP3LM, 'rg_HTDP1LM':rg_HTDP1LM, 'rg_ATDP1LM':rg_ATDP1LM  })


    #54

        #GOAL DIFFERENCE ON 1,3,5 LAST MATCHS 
    #Variable
    #HT/AT Diff
    columns = ['GoalDiff_HT_5lm_PM', 'GoalDiff_AT_5lm_PM', 'GoalDiff_HT_3lm_PM', 'GoalDiff_AT_3lm_PM', 'GoalDiff_HT_1lm_PM',
            'GoalDiff_AT_1lm_PM','HT_Diff_Goal_Diff_5lm', 'AT_Diff_Goal_Diff_5lm', 'HT_Diff_Goal_Diff_3lm',
            'AT_Diff_Goal_Diff_3lm', 'HT_Diff_Goal_Diff_1lm', 'AT_Diff_Goal_Diff_1lm']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_GDHT5LMPM, rg_GDAT5LMPM, rg_GDHT3LMPM, rg_GDAT3LMPM, rg_GDHT1LMPM, rg_GDAT1LMPM, rg_HTDGD5LM, rg_ATDGD5LM, rg_HTDGD3LM, rg_ATDGD3LM, rg_HTDGD1LM, rg_ATDGD1LM = [ds_len - 12, ds_len - 11, ds_len - 10, ds_len - 9, ds_len - 8, ds_len - 7, ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_GDHT5LMPM':rg_GDHT5LMPM, 'rg_GDAT5LMPM':rg_GDAT5LMPM, 'rg_GDHT3LMPM':rg_GDHT3LMPM, 'rg_GDAT3LMPM':rg_GDAT3LMPM, 'rg_GDHT1LMPM':rg_GDHT1LMPM, 'rg_GDAT1LMPM':rg_GDAT1LMPM, 'rg_HTDGD5LM':rg_HTDGD5LM, 'rg_ATDGD5LM':rg_ATDGD5LM, 'rg_HTDGD3LM':rg_HTDGD3LM, 'rg_ATDGD3LM':rg_ATDGD3LM, 'rg_HTDGD1LM':rg_HTDGD1LM, 'rg_ATDGD1LM':rg_ATDGD1LM  })




        #RANKING ON 1,2,5 LAST MATCHS (prematch)
    #Variable
    columns = ["HT_5lm_week_ranking", "AT_5lm_week_ranking", "HT_3lm_week_ranking", "AT_3lm_week_ranking",
            "HT_1lm_week_ranking", "AT_1lm_week_ranking"]
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HT5lm_WR, rg_AT5lm_WR, rg_HT3lm_WR, rg_AT3lm_WR, rg_HT1lm_WR, rg_AT1lm_WR = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HT5lm_WR':rg_HT5lm_WR, 'rg_AT5lm_WR':rg_AT5lm_WR, 'rg_HT3lm_WR':rg_HT3lm_WR, 'rg_AT3lm_WR':rg_AT3lm_WR, 'rg_HT1lm_WR':rg_HT1lm_WR, 'rg_AT1lm_WR':rg_AT1lm_WR })


                                        #STATISTIQUES DE JEU



            #CORNERS NB (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_corners_nb', 'AT_corners_nb', 'HT_Diff_avg_corners_nb', 'AT_Diff_avg_corners_nb']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTCN, rg_ATCN, rg_HTDACN, rg_ATDACN  = [ ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTCN':rg_HTCN, 'rg_ATCN':rg_ATCN, 'rg_HTDACN':rg_HTDACN, 'rg_ATDACN':rg_ATDACN })




            #YELLOW AND RED CARDS NB (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_yellow_cards_nb', 'AT_yellow_cards_nb', 'HT_red_cards_nb', 'AT_red_cards_nb',
            'HT_Diff_avg_yellow_cards_nb', 'AT_Diff_avg_yellow_cards_nb', 'HT_Diff_avg_red_cards_nb',
            'AT_Diff_avg_red_cards_nb']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTYCN, rg_ATYCN, rg_HTRCN, rg_ATRCN, rg_HTDAYCNB, rg_ATDAYCNB,  rg_HTDARCNB,  rg_ATDARCNB  = [ds_len - 8, ds_len -7, ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTYCN':rg_HTYCN, 'rg_ATYCN':rg_ATYCN, 'rg_HTRCN':rg_HTRCN, 'rg_ATRCN':rg_ATRCN, 'rg_HTDAYCNB':rg_HTDAYCNB, 'rg_ATDAYCNB':rg_ATDAYCNB, 'rg_HTDARCNB':rg_HTDARCNB, 'rg_ATDARCNB':rg_ATDARCNB })



            #NB of SHOTS (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg
    #Per match Avg HT/AT Diff
    columns = ['HT_shots_nb', 'AT_shots_nb','HT_avg_shots_nb', 'AT_avg_shots_nb', 'HT_Diff_avg_shots_nb', 'AT_Diff_avg_shots_nb']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTSN, rg_ATSN, rg_HTASN, rg_ATASN, rg_HTDASNB, rg_ATDASNB  = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTSN':rg_HTSN, 'rg_ATSN':rg_ATSN, 'rg_HTASN':rg_HTASN, 'rg_ATASN':rg_ATASN, 'rg_HTDASNB':rg_HTDASNB, 'rg_ATDASNB':rg_ATDASNB})




            #NB of SHOTS ON TARGET (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg
    #Per match Avg HT/AT Diff
    columns = ['HT_shots_on_target_nb', 'AT_shots_on_target_nb', 'HT_avg_shots_on_target_nb', 'AT_avg_shots_on_target_nb',  'HT_Diff_avg_shots_on_target_nb', 'AT_Diff_avg_shots_on_target_nb']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTSOTN, rg_ATSOTN, rg_HTASOTN, rg_ATASOTN, rg_HTDSOTN, rg_ATDSOTN = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTSOTN':rg_HTSOTN, 'rg_ATSOTN':rg_ATSOTN, 'rg_HTASOTN':rg_HTASOTN, 'rg_ATASOTN':rg_ATASOTN, 'rg_HTDSOTN':rg_HTDSOTN, 'rg_ATDSOTN':rg_ATDSOTN})




        #NB of FOULS (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_fouls_nb', 'AT_fouls_nb', 'HT_avg_fouls_nb', 'AT_avg_fouls_nb',  'HT_Diff_avg_fouls_nb', 'AT_Diff_avg_fouls_nb']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTFN, rg_ATFN, rg_HTAFN, rg_ATAFN, rg_HTDFN, rg_ATDFN = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTFN':rg_HTFN, 'rg_ATFN':rg_ATFN, 'rg_HTAFN':rg_HTAFN, 'rg_ATAFN':rg_ATAFN, 'rg_HTDFN':rg_HTDFN, 'rg_ATDFN':rg_ATDFN})




    #POSSESSION (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_possession', 'AT_possession', 'HT_avg_possession', 'AT_avg_possession',  'HT_Diff_avg_possession', 'AT_Diff_avg_possession']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTP, rg_ATP, rg_HTAP, rg_ATAP, rg_HTDP, rg_ATDP = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTP':rg_HTP, 'rg_ATP':rg_ATP, 'rg_HTAP':rg_HTAP, 'rg_ATAP':rg_ATAP, 'rg_HTDP':rg_HTDP, 'rg_ATDP':rg_ATDP})



    #EXPECTED GOALS (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_xg', 'AT_xg', 'HT_avg_xg', 'AT_avg_xg',  'HT_Diff_avg_xg', 'AT_Diff_avg_xg']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTXG, rg_ATXG, rg_HTAXG, rg_ATAXG, rg_HTDXG, rg_ATDXG = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTXG':rg_HTXG, 'rg_ATXG':rg_ATXG, 'rg_HTAXG':rg_HTAXG, 'rg_ATAXG':rg_ATAXG, 'rg_HTDXG':rg_HTDXG, 'rg_ATDXG':rg_ATDXG})




    # 1 / VICTORY ODDS (pre match) (since the beginning of the season)
    #Variable
    #Per match Avg HT/AT Diff
    columns = ['HT_odds_victory_proba', 'AT_odds_victory_proba', 'HT_avg_odds_victory_proba', 'AT_avg_odds_victory_proba',  'HT_Diff_avg_odds_victory_proba', 'AT_Diff_avg_odds_victory_proba']
    temp_df = pd.DataFrame(0, index=dataset_0.index, columns=columns)
    dataset_0 = pd.concat([dataset_0, temp_df], axis=1)
    ds_len = dataset_0.shape[1]
    rg_HTOVP, rg_ATOVP, rg_HTAOVP, rg_ATAOVP, rg_HTDOVP, rg_ATDOVP = [ds_len - 6, ds_len - 5, ds_len - 4, ds_len - 3, ds_len - 2, ds_len - 1]
    
    dico_col_ranks.update({'rg_HTOVP':rg_HTOVP, 'rg_ATOVP':rg_ATOVP, 'rg_HTAOVP':rg_HTAOVP, 'rg_ATAOVP':rg_ATAOVP, 'rg_HTDOVP':rg_HTDOVP, 'rg_ATDOVP':rg_ATDOVP})
    
    
    
    return(dataset_0, dico_col_ranks)


def test_columns_ranks(dico_col_ranks_0, theoritical_df_col_nb, dataset_0):
        """ 
                This function executes a test to verify that the function add_columns_and_complete_col_ranks() was executed correctly on dataset_0. If the tests are successful, it returns True. It checks that:
                - the dataset_0 columns number corresponds to the theoretical value inputted with 'theoritical_df_col_nb'
                - there exists a variable in dico_col_ranks for each new column rank of dataset_0
                - there is not twice the same rank for two different variables in dico_col_ranks
                - there is not twice the same name for two different col ranks in dico_col_ranks
                
        Args:
                dico_col_ranks_0 (Dictionnary): The dictionnary returned by add_columns_and_complete_col_ranks(), that contains new columns ranks.
                
                theoritical_df_col_nb (int): The theoretical columns number in dataset_0
                
                dataset_0 (DataFrame): Dataframe that contains our data, with the new features columns created by add_columns_and_complete_col_ranks().
        
        Returns:
                Boolean : True if the test is passed, False otherwise
        """
        L,l = dataset_0.shape
        list_theoritical_col_rk = [i for i in range(constant_variables.raw_dataframe_col_nb,l)]

        list_values_of_col_rk = []
        list_col_rank_names = []
        
        for key, value in dico_col_ranks_0.items():
                list_values_of_col_rk.append(value)
                list_col_rank_names.append(key)
        
        #Check that the dataframe columns name corresponds to the theoritical value
        dataframe_size_is_good = (l==theoritical_df_col_nb)
        
        #Check that for each column created that there is a col_rank created in the dictionnary
        all_col_has_a_rk = all(valeur in list_values_of_col_rk for valeur in list_theoritical_col_rk)
        
        #Check that there is not twice the same rank for two different variables in dico_col_ranks
        if len(list_values_of_col_rk) == len(set(list_values_of_col_rk)):
                not_twice_same_col_rank = True
        else:
                not_twice_same_col_rank = False
                
        #Check that there is not twice the same name for two different col ranks:
        if len(list_col_rank_names) == len(set(list_col_rank_names)):
                not_twice_same_col_rk_name = True
        else:
                not_twice_same_col_rk_name = False
        
        
        if not_twice_same_col_rk_name and not_twice_same_col_rank and all_col_has_a_rk and dataframe_size_is_good:
                return True
        elif not_twice_same_col_rk_name == False:
                print('not_twice_same_col_rk_name = False')
                return False
        elif not_twice_same_col_rank == False:
                print('not_twice_same_col_rank = False')
                return False
        elif all_col_has_a_rk == False:
                print('all_col_has_a_rk = False')
                return False
        else :
                print('dataframe_size_is_good = False')
                return False
