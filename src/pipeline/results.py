"""This module contains functions used to display the results of our pipeline"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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



# Displaying the results of the model Calibration
def Calibration_results(CalibrationDisplay_non_calibrated, CalibrationDisplay_calibrated, X_valid_0, X_test_0):
     """  
        Present the results of the model calibration. Display 
        - The calibration curves for the model before calibration and after
        - The size of datasets used to train and test the calibrator 
        - A table with the numerical data of the calibration curve of the calibrated model
        - The average deviation of probabilities for our calibrated model.
    
     Args:
        CalibrationDisplay_non_calibrated (sklearn.calibration.CalibrationDisplay): The figure of the calibration curve on X_test of the non calibrated model.
        
        CalibrationDisplay_calibrated (sklearn.calibration.CalibrationDisplay): The figure of the calibration curve on X_test of the calibrated model.
        
        X_valid_0 (DataFrame): The features Dataframe we used to train the calibrator. We use it to display its size (nb of rows).
        
        X_test_0 (DataFrame): The features Dataframe we used to test the calibrated model. We use it to display its size (nb of rows).
    
     Returns:
        None
     """
    
     # Plot the non calibrated model calibration curve
     CalibrationDisplay_non_calibrated.plot()
     # Add labels, legend, and grid
     plt.title('Calibration Curve for non calibrated Pipeline')
     plt.xlabel('Mean Predicted Probability')
     plt.ylabel('Fraction of Positives')
     plt.legend(['Perfectly calibrated', 'Non calibrated pipe calibrat° curve'], loc='best')
     plt.minorticks_on() 
     plt.grid(linewidth=0.5, which ='minor')
     plt.grid(linewidth=0.9)
     plt.show() 


     # Plot the calibrated model calibration curve
     CalibrationDisplay_calibrated.plot()
     # Customize the plot
     plt.xlabel('Mean Predicted Probability')
     plt.ylabel('Fraction of Positives')
     plt.title('Calibration Curve for calibrated pipeline')
     plt.legend(['Perfectly calibrated', 'Calibrated pipe calibrat° curve'], loc='best')
     plt.minorticks_on() 
     plt.grid(linewidth=0.5, which ='minor')
     plt.grid(linewidth=0.9)
     plt.show() 
 
 
     #On calcule et affiche, pour le model calibré, le ratio deviation (moyenne des différence entre prob_pred and prob_true)
     # Calcul des différences terme à terme des proba predicted and true
     differences = np.abs(CalibrationDisplay_calibrated.prob_pred - CalibrationDisplay_calibrated.prob_true)
 
     # Calcul de la moyenne des valeurs absolues des différences qui constitue la deviation
     deviation = np.mean(differences)
 
     # Créer le tableau qui contiendra les ratiaux de deviation ainsi que prob_true
     table_data = {
         'Proba True': np.round(CalibrationDisplay_calibrated.prob_true, decimals=3),
         'Diff with calibrated pred proba': np.round(differences, decimals=3),
     }
 
     calibration_df = pd.DataFrame(table_data)

     #On afffiche la taille du train_set du calibrateur et du test_set sur lequel on a testé notre model calibré
     print('train_set size of the calibrator : ', X_valid_0.shape[0])
     print('test_set size used to test the model-calibrator (nb de matchs sur lequel on a réalisé les calibr curves) : ', X_test_0.shape[0])
 
     # Afficher le tableau 
     print(calibration_df)
 
     print('La deviation moyenne pour ce paramétrage est de ', round(deviation*100, 2), "%")
     print('Le nombre de matchs sur lequel on a réalisé les tests de calibrage est:', X_test_0.shape[0])