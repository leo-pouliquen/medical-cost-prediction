from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from . import data_preprocessing
import joblib
import os
import sys

# Ajout de la racine au path pour importer config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# =============================================================================

# Régression linéaire
def linear_regression(X_train, y_train):
    
    pipeline = data_preprocessing.build_pipeline(LinearRegression())
    pipeline.fit(X_train, y_train)

    print("Régression linéaire entrainée")

    return pipeline

# =============================================================================

# Régression LASSO
def lasso_regression(X_train, y_train):
    lasso_cv = LassoCV(alphas=None, cv= 5, max_iter=10000) # cross-val sur 5 folds
    pipeline = data_preprocessing.build_pipeline(lasso_cv)
    pipeline.fit(X_train, y_train)

    print("Régression LASSO entrainée")
    
    return pipeline

# =============================================================================

# GLM Tweedie
def glm_tweedie_regression(X_train, y_train):

    pipeline = data_preprocessing.build_pipeline(TweedieRegressor(power=1.5, link="log"))
    pipeline.fit(X_train, y_train)

    print("Régression GLM Tweedie entrainée")

    return pipeline

# =============================================================================

# GLM Gamma
def glm_gamma_regression(X_train, y_train):

    pipeline = data_preprocessing.build_pipeline(TweedieRegressor(power=2, link="log"))
    pipeline.fit(X_train, y_train)

    print("Régression GLM Gamma entrainée")

    return pipeline

# =============================================================================

# Random Forest
def random_forest(X_train, y_train):

    pipeline = data_preprocessing.build_pipeline(RandomForestRegressor(n_estimators=100, random_state=42))
    pipeline.fit(X_train, y_train)

    print("Random Forest entrainé")

    return pipeline

# =============================================================================

# XGBoost
def xgboost(X_train, y_train):

    pipeline = data_preprocessing.build_pipeline(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    pipeline.fit(X_train, y_train)

    print("XGBoost entrainé")

    return pipeline

# =============================================================================

# XGBoost optimisé
def optimize_xgboost(X_train, y_train):

    # Grille d'hyperparamètres à tester
    # 3 × 3 × 3 = 27 combinaisons × 5 folds = 135 entraînements au total
    param_grid = {
        'model__n_estimators' : [100, 200, 300],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }

    pipeline = data_preprocessing.build_pipeline(XGBRegressor(random_state=42))

    # GridSearchCV évalue chaque combinaison par validation croisée à 5 folds
    # scoring='r2' pour optimiser le coefficient de détermination
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"Meilleur R² (CV) : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_ 

# =============================================================================

# Fonction de sauvegarde du modèle
def save_model(pipeline, filename):
    # Construction du chemin de sauvegarde depuis la racine du projet    
    filepath = os.path.join(config.base_dir, "models", filename)
    # Création du dossier models/ s'il n'existe pas
    os.makedirs(os.path.join(config.base_dir, "models"), exist_ok=True)
    # Sérialisation du pipeline avec joblib (mieux que pickle pour les modèles sklearn)
    joblib.dump(pipeline, filepath)
    print(f"Modèle sauvegardé : {filepath}")

# =============================================================================

# Fonction de chargement du modèle
def load_model(filename):
    # Chemin de sauvegarde depuis la racine du projet
    filepath = os.path.join(config.base_dir, "models", filename)

    # Vérification que le fichier existe avant le chargement
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modèle introuvable : {filepath}")

    # Chargement du pipeline sérialisé — prêt à prédire sans réentraînement
    pipeline = joblib.load(filepath)
    print(f"Modèle chargé : {filepath}")
    return pipeline