from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# =============================================================================

# Fonction d'évaluation du modèle en fonction du R2, MAE, et RMSE
def evaluate_model(pipeline, X_test, y_test, model_name, log_target=False):
    
    # Génération des prédiction sur le set de test
    y_pred = pipeline.predict(X_test)

    # Pour avoir la même base de comparaison en fonction de la colonne target choisie
    # Back-transformation exp() si le modèle a été entraîné sur log_charges
    if log_target:
        y_pred_eval = np.exp(y_pred)
        y_test_eval = np.exp(y_test)
    else:
        y_pred_eval = y_pred
        y_test_eval = y_test

    
    # Calcul des métriques d'évaluation
    r2 = r2_score(y_test_eval, y_pred_eval)
    mae = mean_absolute_error(y_test_eval, y_pred_eval)
    rmse = root_mean_squared_error(y_test_eval, y_pred_eval)



    print("=" * 50)
    print(f'{model_name}')
    print("=" * 50)
    print(f'Score R² : {r2:.4f}')
    print(f'MAE : {mae:.4f}')
    print(f'RMSE : {rmse:.4f}')
    
    # Retourne un dictionnaire pour alimenter le tableau comparatif
    return {
        "modèle": model_name,
        "R²": round(r2, 4),
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2)
    }

# =============================================================================

# Fonction de stockage dans la liste
def store_and_display(results, metrics):
    results.append(metrics)
    print(f"{len(results)} modèle(s) enregistré(s)")
    df_results = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    return df_results

# =============================================================================

# Fonction de représentation graphique des observations prédites vs observées
def plot_prediction(pipeline, X_test, y_test, model_name):

    # Génération des prédictions pour la visualisation
    y_pred = pipeline.predict(X_test)

    plt.figure()
    sns.kdeplot(y_test, label='Valeurs observées', color= 'royalblue')
    sns.kdeplot(y_pred, label='Valeurs prédites', color= 'red')
    plt.legend(loc='upper right')
    plt.title(f'Comparaison des distributions : réel vs prédit - {model_name}')
    plt.show()

