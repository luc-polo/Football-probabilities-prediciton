"""
This module is made to define variables that keep constant value from the beginning to the end of the programm execution. See below the variables to understand what we talk about.
"""
#Import classic Python modules
from sklearn.model_selection import  StratifiedKFold
import sys

#modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from data.make_dataset import *


#Nb of teams in championship
nb_teams = 20 

#Nb of game weeks in the championship
nb_championship_weeks = 38

# Define the stratified k-fold cross-validation strategy that  we will use in several functions
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#On créé une liste des dates de fin de saison (en réalité on prend une date au milieu de l'inter saison pour etre sur d'etre dans les clous)
seasons = [ "07/15/2015","07/15/2016","07/15/2017","07/15/2018","07/15/2019","07/15/2020","07/15/2021","07/15/2022", "07/15/2023"]

#Raw data dataframe col number
raw_dataframe_col_nb = 66
