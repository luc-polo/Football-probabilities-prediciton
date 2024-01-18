"""This module is made to contain anything necessary to make features selection. 
For instance, the function tracer_feature_mean_depending_on_game_week() is not directly made for this but allows to know from what Game Week we can compute features correlation with matchs' results to have reliable results"""


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import seaborn as sns

import useful_functions
from data import constant_variables

# --------------------------------------------------------------
# Features Correlation with matchs' results computing (used in 'II)1)')
# --------------------------------------------------------------

#Ploting the mean of a feature at every championship day
def plot_feature_stats_over_game_weeks(home_feature_column_name, away_feature_column_name, dataset_0):
    """  
    This function creates a line plot to visualize the mean values of a particular feature across different game weeks. This function is (in my memories), made to know starting what Game Week we should calculate the correlation between features and match results. Indeed, at the beginning of the year the features avg are too weak and unrepresentative.
    
    Args:
        home_feature_column_name (str): Name of the column of the home feature we want to compute the avg over Game Weeks
        
        away_feature_column_name (str): Name of the column of the away feature we want to compute the avg over Game Weeks
        
        
        dataset_0 (DataFrame): The dataset containing the data.
    
    Returns:
        Line2D: Line plot with game weeks on the x-axis and the mean feature values on the y-axis. The plot represents the trend of the mean feature values across different game weeks.
    """
    # Concaténer les colonnes
    Home_Away_Features = pd.concat([dataset_0[home_feature_column_name], dataset_0[away_feature_column_name]], axis=0, ignore_index=True)
    Game_Week = pd.concat([dataset_0['Game Week'], dataset_0['Game Week']], axis=0, ignore_index=True) 
    concatenated_df = pd.DataFrame({'Home_Away_Features': Home_Away_Features,'Game Week': Game_Week})

    x = [i for i in range(1,39)]
    
    y_mean = [concatenated_df[concatenated_df['Game Week'] == week]['Home_Away_Features'].mean() for week in range(1, 39)]
    y_std = [concatenated_df[concatenated_df['Game Week'] == week]['Home_Away_Features'].std() for week in range(1, 39)]
    
    # Création du graphique
    plt.plot(x, y_mean, marker='o', linestyle='-', color='b', label='Mean Home_Away_Features')
    # Ajoutez des étiquettes d'axe et un titre
    plt.xlabel('Game Week')
    plt.ylabel(f'Mean of {home_feature_column_name.replace("HT_", "")}')
    plt.title(f'Mean of {home_feature_column_name.replace("HT_", "")} over Game Weeks')
    plt.grid(True)
    plt.legend()
    # Affichez le graphique
    plt.show()
    
    # Create the standard deviation plot
    plt.plot(x, y_std, marker='o', linestyle='-', color='r', label=f'Standard Deviation of {home_feature_column_name}')
    # Add axis labels and title for the standard deviation plot
    plt.xlabel('Game Week')
    plt.ylabel(f'Standard Deviation of {home_feature_column_name.replace("HT_", "")}')
    plt.title(f'Standard Deviation of {home_feature_column_name.replace("HT_", "")} over Game Weeks')
    # Add grid and legend for the standard deviation plot
    plt.grid(True)
    plt.legend()
    # Show the standard deviation plot
    plt.show()
    
#Assessing the CORRELATION between two features
def calculcate_feature_correlation( home_feature_column_name, away_feature_column_name, k, min_week_game, date_min, dataset_0):
    """  
    This function computes the correlation of biserial point between a specific feature and the matchs' results, applying some filters to select matchs used to compute our result.
    
    Args:
        home_feature_column_name (str): Name of the column of the home feature we want to compute the correlation of biserial point

        away_feature_column_name (str): Name of the column of the away feature we want to compute the correlation of biserial point
        
        k (float): Value used to compute the limits of outliers. We will multiply the IQR by k to compute lower and upper bound for outliers. If we don't want outliers elimination we just need to put a big value for k (like 999).
        
        min_week_game (int): The min Game Week of the match we will select to compt the correlation.
        
        date_min (datetime): If we want to select only the matchs that played out after a certain date, refer the date with this parameter.
        
        dataset_0 (DataFrame): The dataset containing the data.
    
    Returns:
        Tuple: Correlation value, p-value, Mean of the feature when Result = 1, Mean of the feature when Result = 0, df_without_outliers
    """
    
    #Si on a décidé d'enlever les lignes avant une certaine date:
    
    dataset_01 = dataset_0[dataset_0["date_GMT"]>date_min]
    
    #On supprime les lignes qui correspondent à des matchs de journée de championnat < min_week_game:
    dataset_01 = dataset_01[dataset_01["Game Week"]>= min_week_game]
    
    # Concaténer les colonnes
    Home_Away_Features = pd.concat([dataset_01[home_feature_column_name], dataset_01[away_feature_column_name]], axis=0, ignore_index=True)
    RH_RA = pd.concat([dataset_01['RH'], dataset_01['RA']], axis=0, ignore_index=True) 
    concatenated_df = pd.DataFrame({'Home_Away_Features': Home_Away_Features,'RH_RA': RH_RA})

    #On supprime les outliers en calculant les bornes au dela desquelles on considèrent une valeur comme outlier
    #Pour se faire on utilise les quartiless
    Q1 = concatenated_df["Home_Away_Features"].quantile(0.25)
    Q3 = concatenated_df["Home_Away_Features"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    #On supprime les valeurs au dela et en deca des bornes définis à l'aide des quartiles
    df_without_outliers = concatenated_df[(concatenated_df["Home_Away_Features"] >= lower_bound) & (concatenated_df["Home_Away_Features"] <= upper_bound)]
    
    # Calcul de la corrélation entre RH_RA et Home_Away_Feature
    # Calcul de la corrélation de point bisérial spécialement adaptée au cas: variable binaire (1/0) et une variable continue
    correlation, p_value = pointbiserialr(df_without_outliers["RH_RA"], df_without_outliers["Home_Away_Features"])
    
    # Calculer les moyennes de Home_Away_Features pour RH_RA = 1 et RH_RA = 0
    mean_ra1 = df_without_outliers[df_without_outliers['RH_RA'] == 1]['Home_Away_Features'].mean()
    mean_ra0 = df_without_outliers[df_without_outliers['RH_RA'] == 0]['Home_Away_Features'].mean()
    
    return correlation, p_value, mean_ra1, mean_ra0, df_without_outliers

def feature_correlation_with_results_analysis( home_feature_column_name, away_feature_column_name, k, min_week_game, date_min, dataset_0):    
    """  
    This function computes the correlation of biserial point between a specific feature and the matchs' results, applying some filters to select matchs used to compute our result. Other statistics are also displayed: Avg of the feature when Result = 1 and 0. As well as p value.
    
    Args:
        home_feature_column_name (str): Name of the column of the home feature we want to compute the correlation of biserial point

        away_feature_column_name (str): Name of the column of the away feature we want to compute the correlation of biserial point
        
        k (float): Value used to compute the limits of outliers. We will multiply the IQR by k to compute lower and upper bound for outliers. If we don't want outliers elimination we just need to put a big value for k (like 999).
        
        min_week_game (int): The min Game Week of the match we will select to compte the correlation.
        
        date_min (datetime): If we want to select only the matchs that played out after a certain date, refer the date with this parameter.
        
        dataset_0 (DataFrame): 
    
    Returns:
        
    """
    correlation, p_value, mean_ra1, mean_ra0, df_without_outliers_0 = calculcate_feature_correlation( home_feature_column_name, away_feature_column_name, k, min_week_game, date_min, dataset_0)
    
    # Affichage des résultats
    print(f"Corrélation de point bisérial pour {home_feature_column_name}: {correlation:.5f}")
    print(f"Valeur de p : {p_value}")
    print("Moyenne feature pour Resultat = 1: ", mean_ra1)
    print("Moyenne feature pour Resultat = 0: ", mean_ra0)
    
    
    # Création du graphique
    plt.figure(figsize=(12, 6))  # Réglez la taille de la figure
    # Utilisez Seaborn pour un graphique de dispersion (scatter plot) avec une transparence de 0.05
    sns.scatterplot(data = df_without_outliers_0, x='Home_Away_Features', y='RH_RA', alpha=0.013)
    plt.scatter(mean_ra1, 1, color='red', label='Moyenne RH_RA=1')
    plt.scatter(mean_ra0, 0, color='red', label='Moyenne RH_RA=0')
    # Ajoutez des étiquettes d'axe et un titre
    plt.xlabel(home_feature_column_name)
    plt.ylabel('RH_RA')
    plt.title('RH_RA en fonction de home_feature_column_name')
    # Affichez le graphique
    plt.show()
    
def ranking_features_correlation_with_result( liste_features_names_HT, liste_features_names_AT, k, min_week_game, date_min, dataset_0):
    """  
    This function ranks different features on their biseral point correlation. It displays a table with the result of this ranking.
    
    Args:
        liste_features_names_HT (list): Name of the home features we want to compute the correlation of biserial point and to rank.

        liste_features_names_AT (list): Name of the away features we want to compute the correlation of biserial point and to rank.
        
        k (float): Value used to compute the limits of outliers in correlation computing. We will multiply the IQR by k to compute lower and upper bound for outliers. If we don't want outliers elimination we just need to put a big value for k (like 999).
        
        min_week_game (int): The min Game Week of the match we will select to compte the correlation.
        
        date_min (datetime): If we want to select only the matchs that played out after a certain date, refer the date with this parameter.
        
        dataset_0 (DataFrame): Dataframe containing the base data
    
    Returns:
        Table : A table that we will display in the __main__ file using the 'display()' function.
    """
    # On créé un DataFrame vide pour stocker les résultats
    correlation_ranking_DF = pd.DataFrame(columns=['Feature', 'Correlation', 'Feature mean for R = 1', 'Feature mean for R = 0', 'p value'])

    for (home_feature_column_name, away_feature_column_name) in zip(liste_features_names_HT, liste_features_names_AT):
        
        #On calcule la corrélation et les autres statistiques ici
        correlation, p_value, mean_ra1, mean_ra0, variable_inutile = calculcate_feature_correlation( home_feature_column_name, away_feature_column_name, k, min_week_game, date_min, dataset_0)
    
        # Ajoutez les résultats à la DataFrame
        # Créez un DataFrame temporaire pour stocker les résultats de cette itération
        temp_df = pd.DataFrame({
            'Feature': [f'{home_feature_column_name}'],
            'Correlation': [abs(correlation)],
            'Feature mean for R = 1': [mean_ra1],
            'Feature mean for R = 0': [mean_ra0],
            'p value': [p_value],
            'Ecart relatif entre les feature mean for R = 0 ou 1': [abs(mean_ra1-mean_ra0)/mean_ra1]
        })
        # Utilisez pd.concat pour concaténer le DataFrame temporaire avec correlation_ranking_DF
        #Vérifier si le DataFrame est vide avant la concaténation
        if not correlation_ranking_DF.empty:
            # Utilisez pd.concat pour concaténer le DataFrame temporaire avec correlation_ranking_DF
            correlation_ranking_DF = pd.concat([correlation_ranking_DF, temp_df], ignore_index=True)
        else:
            # Si le DataFrame est vide, affectez simplement temp_df à correlation_ranking_DF
            correlation_ranking_DF = temp_df
        

    # Triez le DataFrame par ordre croissant de la valeur de corrélation
    correlation_ranking_DF = correlation_ranking_DF.sort_values(by='Correlation', ascending=False )
    
    #Mise en forme du Dataframe:
    styled_correlation_ranking_DF = correlation_ranking_DF.style.set_table_styles([{'selector': 'th',
                                                                                    'props': [('background-color', 'lightgray'), ('text-align', 'center')]
                                                                                   }]).format({'Correlation': '{:.6f}',
                                                                                               'Feature mean for R = 1': '{:.6f}',
                                                                                               'Feature mean for R = 0': '{:.6f}',
                                                                                               'Ecart relatif entre les feature mean for R = 0 ou 1': '{:.6f}',
                                                                                               'p value': '{:.1e}'})
    # Affichez le DataFrame trié et formaté
    return styled_correlation_ranking_DF

# --------------------------------------------------------------
# Restricted dataset creation (used in 'II)2)')
# --------------------------------------------------------------
def restricted_datasets(dataset_0):
    """ 
        This function returns restricted dataframes that contain subsets of features considered as relevant. We also gather HT and AT features in one column. We will use these DataFrames to explore their data and to test our model with. The function also eliminate the matchs where Game Week < min_played_matchs_nb (defined in constant_variables).
        The function still need to be completed. We must make a shorter subset of features.
    
    Args:
        dataset_0 (Dataframe): The dataframe we want to extract the columns from
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - names_col_concat_rest_ds_2: DataFrame with relevant subsets of 18 features.
            - names_col_concat_rest_ds_3: DataFrame with relevant subsets of 23 features.
        
    """
    #On définit un nb de matchs min joués pour selectionner les lignes du dataset qui seront ajoutées a ces nouveau df
    min_played_matchs_nb_0 = constant_variables.min_played_matchs_nb


    #Restricted dataset 1


    #Restricted dataset 3
    names_col_rest_ds_3 =  ["HT_avg_victory_pm", "AT_avg_victory_pm", 
                            "HT_avg_pm_points_ponderated_by_adversary_perf", "AT_avg_pm_points_ponderated_by_adversary_perf",
                            "HT_avg_goal_diff_pm","AT_avg_goal_diff_pm",
                            "HT_avg_scored_g_conceedded_g_ratio","AT_avg_scored_g_conceedded_g_ratio",
                            "HT_avg_collected_points_pm","AT_avg_collected_points_pm",
                            "HT_week_ranking","AT_week_ranking",
                            "annual budget of HT","annual budget of AT",
                            "Points_HT_5lm_PM","Points_AT_5lm_PM",
                            "GoalDiff_HT_5lm_PM","GoalDiff_AT_5lm_PM",
                            "HT_5lm_week_ranking","AT_5lm_week_ranking",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_avg_shots_nb","AT_avg_shots_nb",
                            "HT_avg_shots_on_target_nb","AT_avg_shots_on_target_nb",
                            "HT_avg_fouls_nb","AT_avg_fouls_nb",
                            "HT_avg_possession","AT_avg_possession",
                            "HT_avg_xg","AT_avg_xg",
                            "HT_avg_odds_victory_proba","AT_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status",
                            "HT_Diff_Points_3lm", "AT_Diff_Points_3lm",
                            "HT_Diff_Goal_Diff_3lm", "AT_Diff_Goal_Diff_3lm",
                            "HT_Diff_Points_1lm", "AT_Diff_Points_1lm",
                            "HT_Diff_Goal_Diff_1lm", "AT_Diff_Goal_Diff_1lm",
                            "HT_Diff_avg_fouls_nb", "AT_Diff_avg_fouls_nb"]
    names_col_concat_rest_ds_3=["Avg_victory",
                                "Avg_points_pm_ponderated_by_adversary_perf",
                                "Avg_goal_diff", 
                                'Avg_scored_g_conceeded_g_ratio',
                                'Avg_collected_points', 
                                'Week_ranking',
                                "Annual_budget", 
                                "Points_5lm", 
                                "Goal_Diff_5lm", 
                                'Week_ranking_5lm', 
                                'Diff_avg_corners_nb',
                                'Avg_shots_nb', 
                                'Avg_shots_on_target_nb', 
                                'Avg_fouls_nb', 
                                'Avg_possession', 
                                'Avg_xg', 
                                'Avg_odds_victory_proba', 
                                'H_A_status', 
                                "Diff_Points_3lm", 
                                "Diff_Goal_Diff_3lm", 
                                "Diff_Points_1lm", 
                                "Diff_Goal_Diff_1lm",
                                "Diff_avg_fouls_nb"]
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_3 = useful_functions.HT_AT_col_merger(names_col_rest_ds_3, names_col_concat_rest_ds_3, min_played_matchs_nb_0, dataset_0)  
                
                          
    #Restricted dataset 2
    names_col_rest_ds_2 = ["HT_avg_victory_pm", "AT_avg_victory_pm", 
                            "HT_avg_pm_points_ponderated_by_adversary_perf", "AT_avg_pm_points_ponderated_by_adversary_perf",
                            "HT_avg_goal_diff_pm","AT_avg_goal_diff_pm",
                            "HT_avg_scored_g_conceedded_g_ratio","AT_avg_scored_g_conceedded_g_ratio",
                            "HT_avg_collected_points_pm","AT_avg_collected_points_pm",
                            "HT_week_ranking","AT_week_ranking",
                            "annual budget of HT","annual budget of AT",
                            "Points_HT_5lm_PM","Points_AT_5lm_PM",
                            "GoalDiff_HT_5lm_PM","GoalDiff_AT_5lm_PM",
                            "HT_5lm_week_ranking","AT_5lm_week_ranking",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_avg_shots_nb","AT_avg_shots_nb",
                            "HT_avg_shots_on_target_nb","AT_avg_shots_on_target_nb",
                            "HT_avg_fouls_nb","AT_avg_fouls_nb",
                            "HT_avg_possession","AT_avg_possession",
                            "HT_avg_xg","AT_avg_xg",
                            "HT_avg_odds_victory_proba","AT_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status"
                            ]
    names_col_concat_rest_ds_2 =["Avg_victory","Avg_points_pm_ponderated_by_adversary_perf","Avg_goal_diff", 
                                'Avg_scored_g_conceeded_g_ratio','Avg_collected_points', 'Week_ranking',
                                "Annual_budget", "Points_5lm", "Goal_Diff_5lm", 'Week_ranking_5lm', 'Diff_avg_corners_nb',
                                'Avg_shots_nb', 'Avg_shots_on_target_nb', 'Avg_fouls_nb', 'Avg_possession', 'Avg_xg', 'Avg_odds_victory_proba',
                                'H_A_status']
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_2 = useful_functions.HT_AT_col_merger(names_col_rest_ds_2, names_col_concat_rest_ds_2, min_played_matchs_nb_0, dataset_0)
        
    return concat_restricted_ds_2, concat_restricted_ds_3
# --------------------------------------------------------------
# Removing outliers (not used but kept in case of...)
# --------------------------------------------------------------
# --------------------------------------------------------------
# Definition of restricted datasets with a selection of revelevant features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Treatment of highly correlated features (not used but kept in case of...)
# --------------------------------------------------------------

# --------------------------------------------------------------
# Wrapper and Filter features selection (not used but kept in case of...)
# --------------------------------------------------------------