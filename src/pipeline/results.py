"""This module contains functions used to display the results of our pipeline"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import calibration_curve
import warnings
from sklearn.utils import check_matplotlib_support, column_or_1d,  check_consistent_length
from sklearn.calibration import _check_pos_label_consistency
import pandas_profiling



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


#We modify the calibration_curve of scikit learn in order it return the width of the bins:
def calibration_curve_bis(
    y_true,
    y_prob,
    *,
    pos_label=None,
    normalize="deprecated",
    n_bins=5,
    strategy="uniform"):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    normalize : bool, default="deprecated"
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.

        .. deprecated:: 1.1
            The normalize argument is deprecated in v1.1 and will be removed in v1.3.
            Explicitly normalizing `y_prob` will reproduce this behavior, but it is
            recommended that a proper probability is used (i.e. a classifier's
            `predict_proba` positive class).

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # TODO(1.3): Remove normalize conditional block.
    if normalize != "deprecated":
        warnings.warn(
            "The normalize argument is deprecated in v1.1 and will be removed in v1.3."
            " Explicitly normalizing y_prob will reproduce this behavior, but it is"
            " recommended that a proper probability is used (i.e. a classifier's"
            " `predict_proba` positive class or `decision_function` output calibrated"
            " with `CalibratedClassifierCV`).",
            FutureWarning,
        )
        if normalize:  # Normalize predicted values into interval [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    
    bin_sums = np.bincount(binids, weights = y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))


    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred, bin_total, bins




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
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    if calibrated_model_or_not == True:
        plt.title('Calibration Curve for calibrated Pipeline')
    elif calibrated_model_or_not == False:
        plt.title("Calibration Curve for non calibrated Pipeline")
    # Afficher nb_bins
    plt.text(0.0, 0.83, f"nb_bins = {n_bins_0}", fontsize=10)

    if calibrated_model_or_not == True:
        plt.legend(['Perfectly calibrated', 'Calibrated pipe calibrat° curve'], loc='best')
    else:
        plt.legend(['Perfectly calibrated', 'Non calibrated pipe calibrat° curve'], loc='best')
    plt.minorticks_on() 
    plt.grid(linewidth=0.5, which ='minor')
    plt.grid(linewidth=0.9)
    plt.show() 
    
    """
    #Plot histogram of predicted probabilities 
    plt.hist(
        Calibration_disp.y_prob,
        range=(0, 1),
        bins= len(Calibration_disp.prob_pred),
        color= color_0)
    plt.show
    """
    
    return Calibration_disp

def plot_calibration_curve_2(pipeline_0, X_test_0, Y_test_0, n_bins_0, strategy_0, color_0, calibrated_model_or_not):
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
    y_proba_pred = pipeline_0.predict_proba(X_test_0)[:,1]
    prob_true, prob_pred, samples_nb_per_bin, bins = calibration_curve_bis(Y_test_0, y_proba_pred, n_bins= n_bins_0, strategy=strategy_0)
    
    prob_pred = [x - 0.09 for x in prob_pred]
    
    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.scatter(prob_pred, prob_true, s = samples_nb_per_bin[samples_nb_per_bin != 0]/3, marker='o', linestyle='-', color= color_0, label='Calibrat° curve')
    plt.plot(prob_pred, prob_true, color= color_0)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probabilities')
    plt.ylabel('Fraction of positives')
    if calibrated_model_or_not == True:
        plt.title('Calibration curve for calibrated pipeline')
    else:
        plt.title('Calibration curve for non calibrated pipeline')
    plt.grid(True)
    
    
    #increasing nb of graduation a grid lines
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    
    #plot the vertical
    for i in range(len(prob_pred)):
        plt.plot([prob_pred[i],prob_true[i]], [prob_true[i], prob_true[i]], linestyle='-', color='black')
        annotation_text = round(abs(prob_pred[i]-prob_true[i]), 2)
        plt.annotate(annotation_text, ((prob_pred[i]+prob_true[i]) / 2, (prob_true[i])), textcoords="offset points", xytext=(0, 10), ha='center')

    # Display the parameter n_bins_0 value
    plt.text(0.00, 0.9, f'n_bins: {n_bins_0}', ha='left', va='top', fontsize=12)
    # Display the number of samples/predictions used to plot the curve
    plt.text(0.00, 0.84, f'test_set size: {X_test_0.shape[0]}', ha='left', va='top', fontsize=12)
    
    plt.legend()
    plt.show()
    
    #Display stats on bins
    print('Statistics on the above learning curve bins:\n')
    learning_curve_bins_stats = pd.DataFrame({
        'Bin interval':[[round(bins[i], 2), round(bins[i+1], 2)]  for i in range(len(bins)-1)],
        'Predictions nb in the bin': [samples_nb_per_bin[i] for i in range(len(bins)-1)]})
    print(learning_curve_bins_stats)
    
    return prob_true, prob_pred

# Print the statistics of the calibrated pipeline
def print_calibration_stats(prob_pred_0, prob_true_0, X_test_0, X_valid_0, calibrated_or_not):
    """
        Present the stats related to the calibrated pipeline calibration:
        - The size of datasets used to train the calibrator and test the pipeline (if it has been calibrated)
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
    differences = np.abs(prob_pred_0 - prob_true_0)
 
    # Calcul de la moyenne des valeurs absolues des différences qui constituent la deviation
    deviation = np.mean(differences)
 
    # Créer le tableau qui contiendra les ratiaux de deviation ainsi que prob_true
    table_data = {
         'Proba True': np.round(prob_true_0, decimals=3),
         f'Diff with {calibrated_or_not} pred proba': np.round(differences, decimals=3),
     }
    
    #Convert table_data into a DataFrame
    calibration_df = pd.DataFrame(table_data)
    
    print('\n\nProbabilities deviation statistics:\n')
    if calibrated_or_not == 'calibrated':
        #On affiche la taille du train_set du calibrateur et du test_set sur lequel on a testé notre model calibré
        print('Train_set size of the calibrator : ', X_valid_0.shape[0])
    
    # Ilprove table design
    fency_table = tabulate(calibration_df, headers='keys', tablefmt='fancy_grid')

    # Afficher le tableau 
    print(fency_table)

    print('\nLa deviation moyenne pour ce paramétrage est de ', round(deviation*100, 2), "%")