
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pipeline import model, results
import numpy as np
from IPython.display import display, HTML

# Create the personnalised pipeline
def create_pipeline(nb_of_feat_to_select=15, Scaler="StandardScaler", penalty='l1', C=1, l1_ratio = 0.5):
    """
    Create a pipeline based on the defined parameters.

    Args:
        nb_of_feat_to_select (int): Number of features to select.
        Scaler (str): Scaler type ('StandardScaler', 'RobustScaler', 'MinMaxScaler').
        penalty (str): Regularization type for Logistic Regression ('l1', 'l2', 'elasticnet', 'None').
        C (float): Regularization strength (inverse of regularization factor).

    Returns:
        Pipeline: Configured machine learning pipeline.
    """
    # Feature selector
    selector = SelectKBest(score_func=f_classif, k=nb_of_feat_to_select)

    # Scaler
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler()
    }
    scaler = scalers.get(Scaler)

    # Logistic Regression
    if penalty == 'None':
        model = LogisticRegression(penalty=None, C=C, solver='saga', max_iter=3000, random_state=999)
    elif penalty == 'elasticnet':
        model = LogisticRegression(penalty=penalty, C=C, l1_ratio=l1_ratio, solver='saga', max_iter=3000, random_state=999)
    else:
        model = LogisticRegression(penalty=penalty, C=C, solver='saga', max_iter=3000, random_state=999)


    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('logistic', model)
    ])

    return pipeline


from sklearn.metrics import log_loss

# Display the personalised pipeline log-loss
def pipe_log_loss(Y_test_00, prob_pred, annual_training):
    Y_test_00 = np.array(Y_test_00).astype(int)  # Conversion en tableau NumPy d'entiers 

    personalised_log_loss = log_loss(Y_test_00, prob_pred)
    if annual_training == True:
        print("Your personnalised pipeline log-loss, annualy trained, is:", personalised_log_loss)

    else:
        print("Your personnalised pipeline log-loss is:                  ", personalised_log_loss)

# Display the best pipeline log-loss
def best_pipe_log_loss():
    proba_pred_best, Y_test_best, _ = results.load_pred_proba("pipeline_pred_proba_and_Y_and_X_info", print_msg =False)
    Y_test_best = Y_test_best.astype(int)
    log_loss_b = log_loss(Y_test_best.values.ravel(), proba_pred_best)
    print("The best pipeline log-loss is:                            ", log_loss_b)

# Plot pipeline calibration curves
def plot_pipeline_pred_results(proba_pred_GW_training, Y_test_GW_training, X_info_GW_training, normal_proba_pred, Y_test, X_test, plot_with_annual_training, best_model_plot=True):
    if plot_with_annual_training == True:
        Y_test_01 = Y_test_GW_training
        X_test_info_01 = X_info_GW_training
        proba_pred = proba_pred_GW_training

    if plot_with_annual_training == False:
        Y_test_01 = Y_test
        X_test_info_01 = X_test
        proba_pred = normal_proba_pred

    #Plot Calibration curve of the pipeline and info about its bins
    prob_pred_01, prob_true_01 = results.plot_calibration_curve_2(
                                    Y_test_0 = Y_test_01.copy(),
                                    X_train_0 = X_test_info_01.copy(),
                                    proba_pred_0 = proba_pred.copy(),
                                    n_bins_0 = 20,
                                    strategy_0 = 'quantile',
                                    color_0 = 'red',
                                    GW_training_or_not = True,
                                    best_model_plot = best_model_plot)

    #We display statistics on the pipeline probabilities deviation 
    results.print_calibration_stats(prob_pred_01.copy(),
                                        prob_true_01.copy(),
                                        calibrated_or_not = 'non calibrated')



# build the table containing the statistics of deviation between BEST proba predicted and bookmakers proba 
def compare_best_pred_proba_and_odds_stats():
    display(HTML('<span style="font-size:24px; font-weight:bold;">Best Pipeline’s Statistics:</span>'))
    proba_pred_best, Y_test_best, X_info_best = results.load_pred_proba("pipeline_pred_proba_and_Y_and_X_info", print_msg =False)
    Y_test_best = Y_test_best.astype(int)
    _, _, odd_proba_pred_proba_compa_dataset_df  = results.compare_pred_proba_and_odds(proba_pred_best.copy() ,X_info_best.copy())
    results.compare_pred_proba_and_odds_stats(odd_proba_pred_proba_compa_dataset_df)
    


