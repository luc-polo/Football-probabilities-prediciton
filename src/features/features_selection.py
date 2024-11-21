"""This module is made to contain anything necessary to make features selection. 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import BaseEstimator, TransformerMixin

import useful_functions
from configuration import constant_variables

# --------------------------------------------------------------
# Features Correlation with matchs' results computing (used in 'III)1)')
# --------------------------------------------------------------
#Ploting the mean of a feature over Game Weeks
def plot_feature_stats_over_game_weeks(home_feature_column_name, away_feature_column_name, dataset_0):
    """  
        This function creates a line plot to visualize the mean values of a particular feature across different game weeks. This function is (in my memories), made to know starting which Game Week the features have a moderated variance. Indeed, at the beginning of the year the features avg are too weak and unrepresentative.
    
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
    plt.ylim(-3, 3)
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
                                                                                    'props': [('background-color', '#404040'),  # Fond plus foncé
                                                                                                ('color', 'white'),  # Texte en blanc
                                                                                                ('text-align', 'center'),
                                                                                                ('font-weight', 'bold'),  # Texte en gras pour l'entête
                                                                                                ('font-size', '14px')]},  # Taille de police légèrement plus grande
                                                                                    {'selector': 'td',
                                                                                    'props': [('text-align', 'center'),  # Centrer le texte dans les cellules
                                                                                                ('font-size', '12px')]}  # Taille de police des données
                                                                                    ]).format({
                                                                                    'Correlation': '{:.6f}',
                                                                                    'Feature mean for R = 1': '{:.6f}',
                                                                                    'Feature mean for R = 0': '{:.6f}',
                                                                                    'Ecart relatif entre les feature mean for R = 0 ou 1': '{:.6f}',
                                                                                    'p value': '{:.1e}'
                                                                                    })
    # Affichez le DataFrame trié et formaté
    return styled_correlation_ranking_DF



def calculcate_feature_f_classif_correlation(home_feature_column_name, away_feature_column_name, min_week_game, date_min, dataset_0):
    """  
        This function computes the f_classif correlation, applying some filters to select matchs used to compute our result.
    
    Args:
        home_feature_column_name (str): Name of the column of the home feature we want to compute the f_classif correlation of

        away_feature_column_name (str): Name of the column of the away feature we want to compute the f_classif correlation of
        
        min_week_game (int): The min Game Week of the match we will select to compute f_classif.
        
        date_min (datetime): If we want to select only the matchs that played out after a certain date, refer the date with this parameter.
        
        dataset_0 (DataFrame): The dataset containing the data.
    
    Returns:
        Tuple: f_classif correlation value, p-value, Mean of the feature when Result = 1, Mean of the feature when Result = 0, df_without_outliers
    """
    
    #Si on a décidé d'enlever les lignes avant une certaine date:
    
    dataset_01 = dataset_0[dataset_0["date_GMT"]>date_min]
    
    #On supprime les lignes qui correspondent à des matchs de journée de championnat < min_week_game:
    dataset_01 = dataset_01[dataset_01["Game Week"]>= min_week_game]
    
    # Concaténer les colonnes
    Home_Away_Features = pd.concat([dataset_01[home_feature_column_name], dataset_01[away_feature_column_name]], axis=0, ignore_index=True)
    RH_RA = pd.concat([dataset_01['RH'], dataset_01['RA']], axis=0, ignore_index=True) 
    concatenated_df = pd.DataFrame({'Home_Away_Features': Home_Away_Features,'RH_RA': RH_RA})
    
    # Calcul de f_classif entre RH_RA et Home_Away_Feature
    f_classif_0, p_value = f_classif(concatenated_df["Home_Away_Features"].values.reshape(-1, 1), concatenated_df["RH_RA"],)
    
    # Calculer les moyennes de Home_Away_Features pour RH_RA = 1 et RH_RA = 0
    mean_ra1 = concatenated_df[concatenated_df['RH_RA'] == 1]['Home_Away_Features'].mean()
    mean_ra0 = concatenated_df[concatenated_df['RH_RA'] == 0]['Home_Away_Features'].mean()
    
    return f_classif_0, p_value, mean_ra1, mean_ra0, concatenated_df

def ranking_features_f_classif( liste_features_names_HT, liste_features_names_AT, min_week_game, date_min, dataset_0):
    """  
        This function ranks different features on their f_classif correlation. It displays a table with the result of this ranking.
    
    Args:
        liste_features_names_HT (list): Name of the home features we want to compute the f_classif correlation and to rank.

        liste_features_names_AT (list): Name of the away features we want to compute the f_classif correlation and to rank.
        
        k (float): Value used to compute the limits of outliers in f_classif computing. We will multiply the IQR by k to compute lower and upper bound for outliers. If we don't want outliers elimination we just need to put a big value for k (like 999).
        
        min_week_game (int): The min Game Week of the match we will select to count the f_classif.
        
        date_min (datetime): If we want to select only the matchs that played out after a certain date, refer the date with this parameter.
        
        dataset_0 (DataFrame): Dataframe containing the base data
    
    Returns:
        Table : A table that we will display in the __main__ file using the 'display()' function.
    """
    # On créé un DataFrame vide pour stocker les résultats
    f_classif_ranking_DF = pd.DataFrame(columns=['Feature', 'f_classif correlation', 'Feature mean for R = 1', 'Feature mean for R = 0', 'p value'])

    for (home_feature_column_name, away_feature_column_name) in zip(liste_features_names_HT, liste_features_names_AT):
        
        #On calcule la corrélation et les autres statistiques ici
        f_classif_0, p_value, mean_ra1, mean_ra0, variable_inutile = calculcate_feature_f_classif_correlation( home_feature_column_name, away_feature_column_name, min_week_game, date_min, dataset_0)

        # Ajoutez les résultats à la DataFrame
        # Créez un DataFrame temporaire pour stocker les résultats de cette itération
        temp_df = pd.DataFrame({
            'Feature': [f'{home_feature_column_name}'],
            'f_classif correlation': [abs(f_classif_0[0])],
            'Feature mean for R = 1': [mean_ra1],
            'Feature mean for R = 0': [mean_ra0],
            'p value': [p_value[0]],
            'Ecart relatif entre les feature mean for R = 0 ou 1': [abs(mean_ra1-mean_ra0)/mean_ra1]
        })
        # Utilisez pd.concat pour concaténer le DataFrame temporaire avec correlation_ranking_DF
        #Vérifier si le DataFrame est vide avant la concaténation
        if not f_classif_ranking_DF.empty:
            # Utilisez pd.concat pour concaténer le DataFrame temporaire avec correlation_ranking_DF
            f_classif_ranking_DF = pd.concat([f_classif_ranking_DF, temp_df], ignore_index=True)
        else:
            # Si le DataFrame est vide, affectez simplement temp_df à correlation_ranking_DF
            f_classif_ranking_DF = temp_df
        
    # Triez le DataFrame par ordre croissant de la valeur de f_classif
    f_classif_ranking_DF = f_classif_ranking_DF.sort_values(by='f_classif correlation', ascending=False )
    #Mise en forme du Dataframe:
    styled_f_classif_ranking_DF = f_classif_ranking_DF.style.set_table_styles([{'selector': 'th',
                                                                                'props': [('background-color', '#404040'),  # Fond gris clair pour les en-têtes
                                                                                            ('text-align', 'center')]},  # Centrer le texte dans les en-têtes
                                                                                {'selector': 'td',
                                                                                'props': [('text-align', 'center'),  # Centrer le texte dans les cellules
                                                                                            ('font-size', '12px')]}  # Taille de police des données
                                                                                ]).format({
                                                                                'f_classif correlation': '{:.6f}',
                                                                                'Feature mean for R = 1': '{:.6f}',
                                                                                'Feature mean for R = 0': '{:.6f}',
                                                                                'Ecart relatif entre les feature mean for R = 0 ou 1': '{:.6f}',
                                                                                'p value': '{:.1e}'
                                                                                })
    # Affichez le DataFrame trié et formaté
    return styled_f_classif_ranking_DF

# --------------------------------------------------------------
# Restricted dataset creation (used in 'III)2)')
# --------------------------------------------------------------
def restricted_datasets(dataset_0):
    """ 
        This function returns restricted dataframes that contain subsets of features considered as relevant. We also gather HT and AT features in one column. We will use these DataFrames to explore their data and to test our model with. The function also eliminates the matchs where Game Week < min_played_matchs_nb (defined in constant_variables).
    
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
    names_col_rest_ds_3 =  ["Diff_HT_avg_victory_pm", "Diff_AT_avg_victory_pm", 
                            "HT_Diff_points_ponderated_by_adversary_perf", "AT_Diff_points_ponderated_by_adversary_perf",
                            "Diff_HT_goal_diff_pm","Diff_HT_goal_diff_pm",
                            "Diff_HT_avg_scored_g_conceded_g_ratio","Diff_AT_avg_scored_g_conceded_g_ratio",
                            "Diff_pnt_HT_ratio","Diff_pnt_AT_ratio",
                            "Diff_HT_ranking","Diff_AT_ranking",
                            "Diff_HT_annual_budget","Diff_AT_annual_budget",
                            "HT_Diff_Points_5lm","AT_Diff_Points_5lm",
                            "Diff_HT_goal_diff_pm","Diff_AT_goal_diff_pm",
                            "Diff_HT_ranking_5lm","Diff_AT_ranking_5lm",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_Diff_avg_shots_nb","AT_Diff_avg_shots_nb",
                            "HT_Diff_avg_shots_on_target_nb","AT_Diff_avg_shots_on_target_nb",
                            "HT_Diff_avg_fouls_nb","AT_Diff_avg_fouls_nb",
                            "HT_Diff_avg_possession","AT_Diff_avg_possession",
                            "AT_Diff_avg_xg","AT_Diff_avg_xg",
                            "HT_Diff_avg_odds_victory_proba","AT_Diff_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status",
                            "HT_Diff_Points_3lm", "AT_Diff_Points_3lm",
                            "HT_Diff_Goal_Diff_3lm", "AT_Diff_Goal_Diff_3lm",
                            "HT_Diff_Points_1lm", "AT_Diff_Points_1lm",
                            "HT_Diff_Goal_Diff_1lm", "AT_Diff_Goal_Diff_1lm",
                            "HT_Diff_avg_fouls_nb", "AT_Diff_avg_fouls_nb",
                            "Season_year", "Season_year"]
    names_col_concat_rest_ds_3=["Diff_Avg_victory",
                                "Diff_Avg_points_pm_ponderated_by_adversary_perf",
                                "Diff_Avg_goal_diff", 
                                'Diff_Avg_scored_g_conceeded_g_ratio',
                                'Diff_Avg_collected_points',
                                'Diff_Week_ranking',
                                "Diff_Annual_budget",
                                "Diff_Points_5lm",
                                "Diff_Goal_Diff_5lm",
                                'Diff_Week_ranking_5lm',
                                'Diff_avg_corners_nb',
                                'Diff_Avg_shots_nb',
                                'Diff_Avg_shots_on_target_nb',
                                'Diff_Avg_fouls_nb',
                                'Diff_Avg_possession',
                                'Diff_Avg_xg',
                                'Diff_Avg_odds_victory_proba',
                                'H_A_status', 
                                "Diff_Points_3lm", 
                                "Diff_Goal_Diff_3lm", 
                                "Diff_Points_1lm", 
                                "Diff_Goal_Diff_1lm",
                                "Diff_avg_fouls_nb",
                                "Season year"]
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_3 = useful_functions.HT_AT_col_merger(names_col_rest_ds_3, names_col_concat_rest_ds_3, min_played_matchs_nb_0, dataset_0)  
                
                          
    #Restricted dataset 2
    names_col_rest_ds_2 = ["Diff_HT_avg_victory_pm", "Diff_AT_avg_victory_pm", 
                            "HT_Diff_points_ponderated_by_adversary_perf", "AT_Diff_points_ponderated_by_adversary_perf",
                            "Diff_HT_goal_diff_pm","Diff_AT_goal_diff_pm",
                            "Diff_HT_avg_scored_g_conceded_g_ratio","Diff_AT_avg_scored_g_conceded_g_ratio",
                            "Diff_pnt_HT_ratio","Diff_pnt_AT_ratio",
                            "Diff_HT_ranking","Diff_AT_ranking",
                            "Diff_HT_annual_budget","Diff_AT_annual_budget",
                            "HT_Diff_Points_5lm","AT_Diff_Points_5lm",
                            "Diff_HT_goal_diff_pm","Diff_AT_goal_diff_pm",
                            "Diff_HT_ranking_5lm","Diff_AT_ranking_5lm",
                            "HT_Diff_avg_corners_nb","AT_Diff_avg_corners_nb",
                            "HT_Diff_avg_shots_nb","AT_Diff_avg_shots_nb",
                            "HT_Diff_avg_shots_on_target_nb","AT_Diff_avg_shots_on_target_nb",
                            "HT_Diff_avg_fouls_nb","AT_Diff_avg_fouls_nb",
                            "HT_Diff_avg_possession","AT_Diff_avg_possession",
                            "HT_Diff_avg_xg","AT_Diff_avg_xg",
                            "HT_Diff_avg_odds_victory_proba","AT_Diff_avg_odds_victory_proba",
                            "HT_H_A_status", "AT_H_A_status", 
                            "Season_year", "Season_year" 
                            ]
    names_col_concat_rest_ds_2 =["Diff_Avg_victory",
                                 "Diff_Avg_points_pm_ponderated_by_adversary_perf",
                                 "Diff_Avg_goal_diff", 
                                 "Diff_Avg_scored_g_conceeded_g_ratio",
                                 "Diff_Avg_collected_points", 
                                 "Diff_Week_ranking",
                                 "Diff_Annual_budget",
                                 "Diff_Points_5lm",
                                 "Diff_Goal_Diff_5lm",
                                 'Diff_Week_ranking_5lm',
                                 'Diff_avg_corners_nb',
                                 'Diff_Avg_shots_nb',
                                 'Diff_Avg_shots_on_target_nb',
                                 'Diff_Avg_fouls_nb',
                                 'Diff_Avg_possession',
                                 'Diff_Avg_xg',
                                 'Diff_Avg_odds_victory_proba',
                                 'H_A_status',
                                 'Season_year']
    #concaténation des colonnes HT et AT dans un meme dataframe
    concat_restricted_ds_2 = useful_functions.HT_AT_col_merger(names_col_rest_ds_2, names_col_concat_rest_ds_2, min_played_matchs_nb_0, dataset_0)
        
    return concat_restricted_ds_2, concat_restricted_ds_3


# --------------------------------------------------------------
#  Removing highly correlated features (not used but kept in case... placed in 'VI))')
# --------------------------------------------------------------
#Transformer à utiliser dans la pipeline elle même
class correlated_features_removal_transformer(BaseEstimator, TransformerMixin):
    """  
        This transformer removes highly correlated features from a dataset. It does this by first calculating the correlation matrix for the dataset and then identifying pairs of features that have a correlation coefficient above a specified threshold. For each pair of highly correlated features, it then removes the feature that has the lowest correlation with the target variable.
        We use it in  the pipeline
    
    Args:
        corr_threshold (float): The corr_threshold parameter determines the threshold at which two features are considered to be highly correlated. Features with a correlation coefficient above this threshold will be removed from the dataset.
    
    Returns:
        dataset without the highly correlated features
        
    """
    def __init__(self, corr_threshold):
        self.corr_threshold = corr_threshold
        
    def fit(self, X_0, Y_0):
        
        #The first step in fit is to find the correlated features in the X_0 dataset 
        #We calculate  the correlation matrix
        X_0 = pd.DataFrame(X_0)
        cor_matrix = X_0.corr().abs()
        # Create an empty list to store the pairs of highly correlated features
        highly_correlated_pairs = []
        #list of features removed from dataframe
        features_to_remove=[]

        # Iterate through the correlation matrix to find highly correlated features
        for i in range(len(cor_matrix.columns)):
            for j in range(i + 1, len(cor_matrix.columns)):
                if cor_matrix.iloc[i, j] > self.corr_threshold:
                    # If the correlation coefficient is above the threshold, add the feature names to the list
                    feature1 = cor_matrix.columns[i]
                    feature2 = cor_matrix.columns[j]
                    highly_correlated_pairs.append((feature1, feature2))

                    #On vérifie qu'on a pas déja placé l'une ou les deux features du dataframe dans la liste des features à retirer
                    if (feature1 not in features_to_remove) and (feature2 not in features_to_remove):
                        #Remove the feature that has the lowest correlation with the target (Y)
                        f_scores1, p_values1 = f_classif(X_0[feature1].to_frame(), Y_0)
                        f_scores2, p_values2 = f_classif(X_0[feature2].to_frame(), Y_0)
                        #On selectionne la feature with the lowest correlation score avec la target
                        if f_scores1 > f_scores2:
                            feature_to_delete = feature2
                        else:
                            feature_to_delete = feature1
                            
                        #On ajoute à "features_to_remove" le nom de la feature à retirer
                        features_to_remove.append(feature_to_delete)
                        
        # Print the highly correlated features pairs
        """for pair in highly_correlated_pairs:
            print(f"Highly correlated features: {pair[0]} and {pair[1]}")"""
        # Print the features removed
        """print("\n Les features choisies pour être retirées du df sont:", features_to_remove)"""
        
        #Second step: register the correlated features names in the features_to_remove variable of the transformer
        #On défini la liste des features que devra retirer le transformer lorsqu'il sera appliqué à un dataset:
        self.features_to_remove = features_to_remove
        
        #On retourne l'objet transformer pour lequel on aura juste défini la liste features_to_remove
        return self
    
    
    def transform(self, X_0):
        # We convert X_0 into a dataframe
        clean_data = pd.DataFrame(X_0)
        #On supprime du clean dataset les features à supprimer
        clean_data.drop(self.features_to_remove, axis=1, inplace=True)
        #We reconvert the cleaned data into a numpy array
        return clean_data.values
    
    
# --------------------------------------------------------------
# Wrapper and Filter features selection (used in 'V)2)')
# --------------------------------------------------------------
def wrapper_features_selection(X_0,Y_0, model_0):
    """  
        Function designed for testing Wrapper feature selection performance and conducting feature selection. It is not included in the pipeline because it significantly slows down its execution. The function displays the performance of model_0, obtained by cross-validation on the training set, using several subsets of features.
    
    Args:
        X_0 (DataFrame):  A DataFrame representing the features of the dataset. Features selection we conduct must be performed on the training set, so X_train should be provided.
        
        Y_0 (DataFrame): A DataFrame representing the target variable of the dataset. (Y_train in our case)
        
        model_0 (model): A machine learning model used to evaluate the importance of the features
    
    Returns:
        Dataframe: X_0 after applying the wrapper feature selection process on it.
        
    """
    sfs = SFS(estimator = model_0, k_features = 'best', forward = True, verbose = 3, cv=4, scoring = 'neg_log_loss')
    sfs.fit(X_0,  np.ravel(Y_0))

    
    #Displaying the detailed results and stats of Sequential Features Selection
    print('Best accuracy score: %.4f' % sfs.k_score_)
    print('Best subset (indices):', sfs.k_feature_idx_)
    print('Best subset (corresponding names):', sfs.k_feature_names_)
    metric_dict = sfs.get_metric_dict(confidence_interval=0.95)
    fig_perf_nb_of_feat = plot_sfs(metric_dict, kind= 'std_dev')
    plt.title('Sequential Selection (w. StdDev)')
    plt.grid()
    plt.show()

    #Adapting X to the features selection made 
    X_wrapped = sfs.transform(X_0)
    #On retrensforme X_wrapped en DataFrame
    X_wrapped = pd.DataFrame(X_wrapped, columns=sfs.k_feature_names_)
    
    return(X_wrapped)


def filter_features_selection(X_0, Y_0, nb_features_to_select, score_func, report):
    """  
        Funciton made to make tests on Filter features selection performances. It displays the performances of 
    
    Args:
        X_0 (DataFrame):  A DataFrame representing the features of the dataset. The test we make must be done on train set so here we must input X_train
        
        Y_0 (DataFrame): A DataFrame representing the target variable of the dataset (Y_train in our case)
        
        nb_features_to_select (int or str): Number of features to select using the filtering method. It can be an integer specifying the exact number of features or a string such as 'all' to select all features.
        
        score_func (func): A scoring function used to evaluate the importance of features, for instance 'f_classif'. This funct° is supposed to computes the correlation (or somthing like that) between a feature col and a label col.
        
        report(Boolean): Wether we want to get a report of se data selection we have just made
    
    Returns:
        DataFrame: X_0 after applying the filter feature selection process on it.
        
    """
    selector = SelectKBest(score_func, k= nb_features_to_select)
    
    #On définit X_filtered pour ne pas perdre le nom des colonnes de X_0
    X_filtered = X_0
    X_filtered = selector.fit_transform(X_filtered, np.ravel(Y_0))
    
    #On obtient les noms des features selectionnées
    selected_features=selector.get_support()
    features_names = X_0.columns
    selected_features_names = features_names[selected_features]
    
    if report == True:
        print('\n \n Filter features Selection Results')
        # Obtenez les scores de toutes les features testées
        feature_scores = selector.scores_
        # Créez un DataFrame pour afficher les résultats
        features_with_scores = pd.DataFrame({'Feature': features_names, 'Score': feature_scores})
        features_with_scores = features_with_scores.sort_values(by='Score', ascending=False, ignore_index = True)
        # Afficher les features sélectionnées et leurs scores
        print(f'Here are the {nb_features_to_select} selected features and their scores')
        print(features_with_scores.head(nb_features_to_select))
        
    #On retrensforme X_RobustScaled en DataFrame
    X_filtered = pd.DataFrame(X_filtered, columns=selected_features_names)
        
        
    return X_filtered
   

