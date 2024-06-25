import mlflow
import mlflow.sklearn
import pandas as pd
import gc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, Lars, ElasticNet, 
                                  BayesianRidge, HuberRegressor, SGDRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge


# Chargement des données (à adapter selon votre structure de données)
def load_data():
    # Exemple : chargement depuis un fichier CSV
    df = pd.read_csv('preprocessed.csv', index_col=0)
    return df

# Préparation des données
def prepare_data(df):
    # Séparation des features et de la cible
    X = df.drop(['rental_price_per_day'], axis=1)
    y = df['rental_price_per_day']
    return X, y


# Définition des modèles à tester
models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'SVR': SVR(),
    'SGDRegressor': SGDRegressor(),
    'HuberRegressor': HuberRegressor(),
    'BayesianRidge': BayesianRidge(),
    'ElasticNet': ElasticNet(),
    'Lars': Lars(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'LinearRegression': LinearRegression(),
}

# Paramètres pour la grille de recherche (à adapter selon vos besoins)
param_grid = {
    'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']},
    'DecisionTreeRegressor': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'SVR': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'poly', 'rbf']},
    'SGDRegressor': {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l2', 'l1', 'elasticnet']},
    'HuberRegressor': {'epsilon': [1.35, 1.5, 1.75, 2.0]},
    'BayesianRidge': {'alpha_1': [1e-6, 1e-3, 1e-1], 'alpha_2': [1e-6, 1e-3, 1e-1]},
    'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
    'Lars': {'n_nonzero_coefs': [100, 200, 300, 500], 'eps': [2.220446049250313e-16, 1e-10, 1e-8, 1e-6]},
    'Lasso': {'alpha': [0.1, 1.0, 10.0]},
    'Ridge': {'alpha': [0.1, 1.0, 10.0]},
    'LinearRegression': {},  # Pas d'hyperparamètres à ajuster pour LinearRegression
}

# Fonction d'entraînement et d'évaluation des modèles
def train_eval_model(model_name, model, X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("https://get-around-mlflow-54d4e77f0f13.herokuapp.com/")
    with mlflow.start_run(run_name=model_name):
        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédiction sur le jeu de test
        y_pred = model.predict(X_test)

        # Calcul des métriques
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Enregistrement des métriques dans MLFlow
        mlflow.log_param('model', model_name)
        mlflow.log_metrics({'mse': mse, 'mae': mae, 'r2': r2})

        # Enregistrement du modèle
        mlflow.sklearn.log_model(model, 'model')

        print(f'{model_name} - MSE: {mse}, MAE: {mae}, R^2: {r2}')


# Chargement des données
df = load_data()

# Préparation des données
X, y = prepare_data(df)

# Séparation des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Boucle sur les modèles à tester
for model_name, model in models.items():
    # Recherche des meilleurs hyperparamètres avec GridSearchCV si nécessaire
    if model_name in param_grid:
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model

    # Entraînement et évaluation du modèle
    train_eval_model(model_name, best_model, X_train, X_test, y_train, y_test)
    gc.collect()