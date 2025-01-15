"""
This module aims at generating graphs and statistics to describe data (raw one as well as the new features created).
Its principal goal is to verify that the data is accurate and that we can use it to feed our model or to create new features.
We use the functions below in:
I)1) and II)4)5)
"""


#Import classic Python modules
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import seaborn as sns
import numpy as np
import sys
import os


# --------------------------------------------------------------
# Plot single feature histogram (used in 'I)1)')
# --------------------------------------------------------------

def plot_single_col_histo(col_name, dataset):  
    """
        This function plots the histogram of a particular feature

    Args:
        col_name (str): Name of the column that contains the feature we want to plot the histogram
        
        dataset (DataFrame): The dataframe that contains our data.

    Returns:
        Line2D: histogram of the feature

    Examples:
        >>> plot_single_col_histo('home_team_yellow_cards')
        Lined2D
    """
    plt.hist(dataset[col_name].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all numerical features histograms (used in 'I)1)' )
# --------------------------------------------------------------

def plot_all_num_features(dataset, save, density_estimate):
    """
        This function plots the histogram of all numerical features contained in the raw dataset, excepted the ones that are not relevant to plot: Date and Game Week. It is plotted in a subplot.

    Args:
        dataset (DataFrame): name of the raw dataframe we want to plot the histogram of
        
        save (Boolean): whether we want to save the histograms produced in reports/figures or not. If True, the function won't display anything and will simply save the histograms in the above location. If set to False it displays the histograms and do not save them.
        
        density_estimate (Boolean): Wether we want to plot estimations of the density curves of features rather than their histograms

    Returns:
        Axes: Subplots with the features histograms
    """
    
    #We select the numerical features to plot
    numerical_features= dataset.select_dtypes(include=['int64', 'float64']).columns
    numerical_features.tolist()

    #We delete date and game week in this list to plot the list features histo:
    columns_to_remove = ['timestamp', 'Game Week']
    numerical_features = [feature for feature in numerical_features if feature not in columns_to_remove]


    # Create subplots
    fig, axes = plt.subplots(nrows=int(len(numerical_features) / 2), ncols=2, figsize=(24, (len(numerical_features))*5))

    # Plot histograms for each numerical feature (excepted timestamp usefull below)
    for i, feature in enumerate(numerical_features):
        if density_estimate == True:
            sns.kdeplot(dataset[feature], color='blue', fill=True, ax=axes[i//2, i%2])
        if density_estimate == False:
            sns.histplot(dataset[feature], bins=20, color='blue', alpha=0.7, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(feature, fontsize=22)
        axes[i//2, i%2].set_xlabel('Values')
        axes[i//2, i%2].set_ylabel('Frequency')
    
    
    if save == False:
        #Adjust Layout setting
        plt.tight_layout(pad=0)
        plt.show()
    
    if save == True and density_estimate == True:
        plt.savefig('../../reports/figures/all_features_density_estimate.png')
    if save == True and density_estimate == False:
        plt.savefig('../../reports/figures/all_features_histo.png')


# --------------------------------------------------------------
# Getting more info on specific features potential outliers
# --------------------------------------------------------------
def count_0_values(dataset, feature, min_game_week, HT_or_AT, seasons_end_date):
    """
        Displays statistics per season, on the 0 values of a specific feature. There is the possibility to apply filters on the 0 values we want to identify statistics. (Like if we want to display only statistics on the lines with 0 values for the specific feature, where game week > 5)

    Args:
        dataset (DataFrame): name of the dataframe we want to identify the matchs/lines in.
        
        feature (str): name of the feature we want to identify the 0 values statistics.
        
        min_game_week (int): Filter of minimum Game Week. | Minimum value of Game Week for the 0 values lines, we will select to calculate statistics from
        
        HT_or_AT (str): Allows to inform the function that the feature we selected is a feature for home teams or away teams. Its value is 'HT' or 'AT'.
        
        seasons_end_date (list): Filter of season. List with the end dates of the seasons we want select to identify the 0 values statistics from.

    Returns:
        (str): A text that gives for each season:
        -Number of 0 values for the specific feature
        -The proportion these 0 values represent in the whole season (nb of 0 values/nb of matches in this season)
        -The name of each team that has 0 values this season
        -The game weeks the teams had the 0 values
        
        For sure these statistics only concern the values that respect the filters set in parameters.
        
    Examples:
        >>> seasons_end_dates = [ "07/15/2015","07/15/2016","07/15/2017","07/15/2018","07/15/2019","07/15/2020","07/15/2021","07/15/2022", "07/15/2023"]
        >>> seasons_end_dates = pd.to_datetime(seasons_end_dates)
        >>> count_0_values( 'Pre-Match PPG (Away)', 5,'AT', seasons_end_dates)
        
        Statistics on 0 values for season  2015 

        Number of values == 0 : 5 
        These values represent 1.52 % of the cases 

        Reims  has  1  '0' values for this feature
        it concerns game weeks: [6]
        Olympique Lyonnais  has  1  '0' values for this feature
        it concerns game weeks: [6]
        Toulouse  has  1  '0' values for this feature
        it concerns game weeks: [7]
        Thonon Evian FC  has  2  '0' values for this feature
        it concerns game weeks: [6 8]


        Statistics on 0 values for season  2016 

        Number of values == 0 : 4 
        These values represent 1.21 % of the cases 

        Troyes  has  2  '0' values for this feature
        it concerns game weeks: [6 8]
        Toulouse  has  1  '0' values for this feature
        it concerns game weeks: [6]
        Olympique Marseille  has  1  '0' values for this feature
        ...
        Ajaccio  has  1  '0' values for this feature
        it concerns game weeks: [8]
    """
    
    for season in seasons_end_date:
        
        print('Statistics on 0 values for season \033[1m\033[91m', season.year, '\033[0m\n' )
        
        rows_where_feat_0 = dataset[(dataset['Game Week']>min_game_week) &
                              (dataset[feature] == 0) &
                              (dataset['date_GMT']< season) &
                              (dataset['date_GMT']> season - relativedelta(years=1))]
        
        nb_0_values = rows_where_feat_0.shape[0]
        
        print('Number of values == 0 :', nb_0_values, '')
        print('These values represent', round((nb_0_values/(380 - min_game_week*10))*100, 2), '% of the cases \n')
        
        #We collect the name of teams that present this statistic
        if HT_or_AT=='HT':
            teams=rows_where_feat_0['home_team_name']
            
        if HT_or_AT=='AT':
            teams=rows_where_feat_0['away_team_name']

        
        for team in set(teams):
            #We collect games weeks where team got feature == 0:
            game_weeks_team = []
            if HT_or_AT == 'HT':
                game_weeks_team = rows_where_feat_0[rows_where_feat_0['home_team_name'] == team]['Game Week'].unique()
            if HT_or_AT == 'AT':
                game_weeks_team = rows_where_feat_0[rows_where_feat_0['away_team_name'] == team]['Game Week'].unique()
            
            
            print('\033[91m',team, '\033[0m has ', teams[teams == team].count(), ' \'0\' values for this feature')
            print('it concerns game weeks:', game_weeks_team)
        print('\n')


# Display matchs where feature has a certain value
def display_matchs_specific_feat_val(dataset, feature, value_to_identify ):
     """
        This function identifies the lines/matchs in dataset, where the feature inputed in parameters, is equal to a specific value, also inputed in parameters.

     Args:
        dataset (DataFrame): Name of the DataFrame we want to make the analysis/indetification on.
        
        feature (str): Name of the feature we want to identify the specific value.
        
        value_to_identify (int): The specific value of 'feature', we want to identify in the dataframe.

     Returns:
        str: Text that gives:
        -Number of matchs/lines that have feature == value_to_identify
        -The Date, teams of matches/lines concerned
    """
     rows_where_feat_neag = dataset[dataset[feature]==value_to_identify]
     nb_neg_val = rows_where_feat_neag.shape[0]
    
     list_matchs_concerned = []
     for index, match in rows_where_feat_neag.iterrows():
        list_matchs_concerned.append(str(match['date_GMT']) +
                                     '  ' +
                                     match['home_team_name'] +
                                     ' - ' +
                                     match['away_team_name'])
    
     print('The number of matchs where', feature, 'is =', value_to_identify, 'is ', nb_neg_val)

     print('Here is the list of the matchs concerned:', list_matchs_concerned)



# --------------------------------------------------------------
# Plot boxplot used in 'II)4)')
# --------------------------------------------------------------

def boxplot(restricted_dataset_0):
    """
        This function plots the boxplots of all the columns in the dataset inputed in parameter. We use it to identify outliers.

    Args:
        restricted_dataset_0 (DataFrame): dataset with HT and AT columns merged, containing a restricted subset of features.

    Returns:
        None
    """
    #On récupère la liste des colonnes de restricted_dataset_0
    col_to_plot_list_0 = restricted_dataset_0.columns
    
    #Boxplot plotting settings and def 
    red_circle = dict(markerfacecolor = 'red', marker = 'o', markeredgecolor = 'white', alpha = 0.1)    
    if len(col_to_plot_list_0) >20 and len(col_to_plot_list_0) <=24:
        fig, axs = plt.subplots(6, 4, figsize =(20,60))
    if len(col_to_plot_list_0) >16 and len(col_to_plot_list_0) <=20:
        fig, axs = plt.subplots(5, 4, figsize =(20,60))

    #Filling the figure with the boxplots of the concat_restricted_ds_2 dataframe
    for i, ax in enumerate(axs.flat):
        if i >= (len(col_to_plot_list_0)):
            break
        
        # Create the boxplot with a specific displaying of outliers points (red_circle) and the mean displayed as a line
        box=ax.boxplot(restricted_dataset_0[col_to_plot_list_0[i]], whis=1.5, flierprops = red_circle, showmeans=True, meanline=True)
        # Add a legend specifying the median and mean lines
        ax.legend([box["medians"][0], box["means"][0]], ['Median', 'Mean'])
        # Set the title for the boxplot
        ax.set_title(col_to_plot_list_0[i])


# --------------------------------------------------------------
# Plot heatmap of features correlation
# --------------------------------------------------------------

def heat_map(restricted_dataset_0):
    """
        This function plots the heatmap of correlation between features in the dataset inputed in parameter. We use it to identify features too much correlated.

    Args:
        restricted_dataset_0 (DataFrame): dataset with HT and AT columns merged, containing a restricted subset of features. If too big the execution will be extremly long.

    Returns:
        None
    """
    #On trace la heatmap entre les features:
    plt.figure(figsize=(10, 8), dpi=500)
    sns.heatmap(restricted_dataset_0.corr(), annot = True, fmt= '.2f')
    plt.show()





# --------------------------------------------------------------
# Status of the study of each feature
# --------------------------------------------------------------
"""
attendence :                         OK
Pre-Match PPG (Home/Away):           not used but there are errors
home/away_ppg:                       not used 
home/away_team_goal_count:           OK
total_goal_count:                    seems OK
total_goals_at_half_time:            seems OK
home/away_team_goal_count_half_time: seems OK
home/away_team_corner_count:         OK
home/away_team_yellow_cards:         OK
home/away_team_red_cards:            seems OK
home/away_team_first_half_cards:     not used
home/away_team_second_half_cards:    not used
home/away_team_shots:                not calculated in the same way as ligue1 official stat. however both have approximattly the same tendency and values stat tends to be undermined
home/away_team_shots_on_target:      not calculated in the same way as ligue1 official stat. however both have approximattly the same tendency and values stat tends to be undermined
home/away_team_shots_off_target:     not used
home/away_team_fouls:                not calculated in the same way as ligue1 official stat. however both have approximattly the same tendency and values stat tends to be undermined
home/away_team_possession:           not calculated in the same way as ligue1 official stat. however both have approximattly the same tendency and values accurate for recent seasons
Home/Away Team pre-Match xG:         Started to be reported during the season 22-23. THere is this col only in the csv files of 22-23, 14-15 and 15-6. However, the colum is empty (full of 0 and NaN) for the 14-15 and 15-16 files. Despite the fact that this col does not exist in most of csv files, the concatenation is well made and a column of NaN values is created for the seasons dataframes missing this col.
team_a/b_xg:                         A lot of 0 values. xg started to be reported starting the season 17-18
average_goals_per_match_pre_match:   not used and i do not know what it is
btts_percentage_pre_match:           (=average BTTS % between both teams) This stat is reported only starting the 3rd Game Week the calculation method is not clear. For the first values Game Week = 3, the values are restricted to 0, 50, 100% it's not accurate Mail send to footystat pending.
over_15_percentage_pre_match:        Same problem as btts_percentage
over_25_percentage_pre_match:        Same problem as btts_percentage
over_35_percentage_pre_match:        Same problem as btts_percentage
over_45_percentage_pre_match:        Same problem as btts_percentage
over_15_HT_FHG_percentage_pre_match: Same problem as btts_percentage
over_05_HT_FHG_percentage_pre_match: Same problem as btts_percentage
over_15_2HG_percentage_pre_match:    Same problem as btts_percentage
over_05_2HG_percentage_pre_match:    Same problem as btts_percentage
average_corner_per_match_prematch:   not used but only calculated starting 3rd Game Week. we do not know precisely how its computed
average_card_per_match_pre_match:    Same problem as average_corner
odds_ft_home_team_win:               not studied
odds_ft_draw:                        not studied
odds_ft_away_team_win:               not studied
odds_ft_over_15:                     not studied
odds_ft_over_25:                     not studied
odds_ft_over_35:                     not studied
odds_ft_over_45:                     not studied
odds_btts_no:                        not studied
odds_btts_yes:                       not studied
"""


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------