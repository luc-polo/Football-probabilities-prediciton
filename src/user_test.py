
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression

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




test_seasons = [2021,2022,2023,2024]