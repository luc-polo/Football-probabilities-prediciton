"""This module contains functions used to display the results of our pipeline"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay

# --------------------------------------------------------------
# GridSearchCV() Results
# --------------------------------------------------------------


# Displaying the results of the GridSearchCV() function that optimised pipeline parameters
def GridSearchCV_results(grid_search_0, X_train_0):
    """  
        Present the results of GridSearchCV() on our pipeline. Display the optimal parameters, score, and features selected by GridSearchCV() on our pipeline.
    
    Args:
        grid_search_0 (object): Name of the GridSearchCV() object ran before and that we want to display the results of.
        
        X_train_0 (DataFrame): The trainset on which we ran grid_search_0 before.
    
    Returns:
        None
    """
    # Display the best parameters and score
    best_params = grid_search_0.best_params_
    best_score = grid_search_0.best_score_
    print("Best Parameters:", best_params, "\n\nBest score with these hyper parameters:", best_score)

    #Display features selected
    # Obtenir le sélecteur de caractéristiques à partir du meilleur estimateur
    best_selector = grid_search_0.best_estimator_['features_selector']
    # Obtenir les indices des features sél"ectionnées
    selected_feature_indices = best_selector.get_support(indices=True)
    # Obtenir les noms des caractéristiques à partir des indices
    selected_feature_names = X_train_0.columns[selected_feature_indices]

    print("\n\nFeatures selected:",selected_feature_names)


# --------------------------------------------------------------
# Calibration Results
# --------------------------------------------------------------

# Plot calibration curves of calibrated and not calibrated pipeline/model
def plot_calibration_curve(pipeline_0, X_test_0, Y_test_0, n_bins_0, strategy_0, color_0, calibrated_model_or_not):
    """  
        Display the annotated calibration curves either for the non calibrated pipeline or for the calibrated one.
     
     Args:
        pipeline_0 (pipeline): The calibrated or non calibrated pipeline we want to plot the calibration curve of.
        
        X_test_0 (DataFrame): The features Dataframe used to plot the calibration curve.
        
        Y_test_0 (DataFrame): The labels/targets Dataframe used to plot the calibration curve.
        
        n_bins_0 (int): Number of bins to discretize the predicted probabilities.
        
        strategy_0 (str): strategy to discretize the probabilities interval to define the bins intervals. Etither 'uniform' or 'quantile'
        
        color_0 (str): Color for plotting the calibration curve. Blue for the calibrated model and Red for non calibrated one.
        
        calibrated_model_or_not (Boolean): Wether the pipeline inputed is the calibrated one or not (used to define graph anotations)
    
     Returns:
        sklearn.calibration.CalibrationDisplay : The figure of calibration curve of pipeline_0
     """
    Calibration_disp = CalibrationDisplay.from_estimator(pipeline_0, X_test_0, Y_test_0, n_bins = n_bins_0, strategy =strategy_0, color= color_0)
    # Add labels, legend, and grid
    if calibrated_model_or_not == True:
        plt.title('Calibration Curve for calibrated Pipeline')
    else:
        plt.title('Calibration Curve for non calibrated Pipeline')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    # Afficher nb_bins
    plt.text(0.0, 0.83, f"nb_bins = {n_bins_0}", fontsize=12)

    if calibrated_model_or_not == True:
        plt.legend(['Perfectly calibrated', 'Calibrated pipe calibrat° curve'], loc='best')
    else:
        plt.legend(['Perfectly calibrated', 'Non calibrated pipe calibrat° curve'], loc='best')
    plt.minorticks_on() 
    plt.grid(linewidth=0.5, which ='minor')
    plt.grid(linewidth=0.9)
    plt.show() 
    
    return Calibration_disp

# Print the statistics of the calibrated pipeline
def print_calibration_stats(CalibrationDisplay_0, X_test_0, X_valid_0):
    """
        Present the stats related to the calibrated pipeline calibration:
        - The size of datasets used to train the calibrator and test the pipeline
        - A table with the numerical data of the pipeline calibration curve (The coordinates of the points on the graph) 
        - The average deviation of probabilities for our pipeline.
    
    Args:
        CalibrationDisplay_0 (sklearn.calibration.CalibrationDisplay): The figure of the calibration curve on X_test of the calibrated pipeline.
        
        X_valid_0 (DataFrame): The features Dataframe we used to train the calibrator. We use it to display its size (nb of rows).
        
        X_test_0 (DataFrame): The features Dataframe we used to test the calibrator. We use it to display its size (nb of rows)
    
    Returns:
        None
    """
    #On calcule et affiche, pour le model calibré, le ratio deviation (moyenne des différence entre prob_pred and prob_true)
    # Calcul des différences terme à terme des proba predicted and true
    differences = np.abs(CalibrationDisplay_0.prob_pred - CalibrationDisplay_0.prob_true)
 
    # Calcul de la moyenne des valeurs absolues des différences qui constituent la deviation
    deviation = np.mean(differences)
 
    # Créer le tableau qui contiendra les ratiaux de deviation ainsi que prob_true
    table_data = {
         'Proba True': np.round(CalibrationDisplay_0.prob_true, decimals=3),
         'Diff with calibrated pred proba': np.round(differences, decimals=3),
     }
    
    #Convert table_data into a DataFrame
    calibration_df = pd.DataFrame(table_data)
    
    # Trier le tableau par ordre décroissant selon la colonne Proba True
    calibration_df = calibration_df.sort_values(by='Proba True', ascending=False)

    #On affiche la taille du train_set du calibrateur et du test_set sur lequel on a testé notre model calibré
    print('Train_set size of the calibrator : ', X_valid_0.shape[0])
    print('Test_set size used to test the pipeline-calibrator (nb de matchs sur lequel on a réalisé les calibr curves) : ', X_test_0.shape[0], '\n')

    # Afficher le tableau 
    print(calibration_df)

    print('\nLa deviation moyenne pour ce paramétrage est de ', round(deviation*100, 2), "%")
    print('Le nombre de matchs sur lequel on a réalisé les tests de calibrage est:', X_test_0.shape[0])