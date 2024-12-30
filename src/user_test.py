
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pipeline import model, results


# Create the personnalised pipeline
def create_pipeline(nb_of_feat_to_select=15, Scaler="StandardScaler", penalty='l1', C=1):
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
    scaler = scalers.get(Scaler, StandardScaler())

    # Logistic Regression
    model = LogisticRegression(penalty=penalty, C=C, solver='saga', max_iter=3000, random_state=999)

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('logistic', model)
    ])

    return pipeline



# Plot pipeline perf
def plot_pipeline_pred_results(proba_pred_GW_training, Y_test_GW_training, X_info_GW_training, normal_proba_pred, plot_with_annual_training, best_model_plot=True):
    if plot_with_annual_training == True:

        Y_test_01 = Y_test_GW_training
        X_test_info_01 = X_info_GW_training
        proba_pred = proba_pred_GW_training

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
                                        prob_true_01.copy())
    if plot_with_annual_training == False: