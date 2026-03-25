import pandas as pd
import config


# Fonction de contrôle des valeurs manquantes
def check_missing_values(df):
    # Comptage du nombre de valeurs manquantes
    missing_values_count = df.isna().sum()
    # Calcul du pourcentage de valeurs manquantes
    missing_percentage = (missing_values_count / len(df) * 100).round(2)


    # On filtre les valeurs manquantes dans un dataframe 
    missing_values = pd.DataFrame({
        "Count" : missing_values_count,
        "Percentage" : missing_percentage
        }).query("Count > 0")

    # Affichage du résultat du test
    if missing_values.empty:
        print("Aucune valeurs manquantes")
    else :
        print("Valeurs manquantes :")
        print(missing_values)

# =============================================================================

# Fonction de contrôle des types de données
# L'objectif est de s'assurer que les types de données sont bien conformes à ceux attendus et définis dans config.py
def check_dtypes(df):
    # Liste pour stocker les erreurs détectées
    errors = []

    # Comparaison des types réels aux types attendus dans config.py
    for col, expected_dtypes in config.expected_dtypes.items():
        df_dtypes = str(df[col].dtype)
        
        # Si le type ne correspond pas, on enregistre l'erreur dans le dictionnaire
        if df_dtypes != expected_dtypes:
            errors.append({
            "colonne" : col,
                "type attendu" : expected_dtypes,
                "type réel" : df_dtypes
            })

    # Affichage du résultat du test
    if not errors:
            print("Tous les types sont conformes")
        
    else :
        print("Attention types non conformes:")
        print(pd.DataFrame(errors))

# =============================================================================

# Fonction de contrôle des doublons
def check_duplicates(df):
    # Comptage du nombre de lignes identiques sur toutes les colonnes
    n_duplicates = df.duplicated().sum()

    # Affichage du résultat du test
    if n_duplicates ==0:
        print('Aucun doublon')
    else:
        print(f"{n_duplicates} doublons détectés")

# =============================================================================

# Fonction de contrôle des outliers
# Nous utilisons l'écart interquartile entre le 3ème QT et le 1er QT
def check_outliers(df):
    # Sélection des colonnes numériques
    num_cols = df.select_dtypes(include="number").columns
    outliers_found = False

    # Calcul des quartiles et de l'écart interquartile
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Définition du seuil : 3 IQR
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Comptage des valeurs hors seuil
        n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

        # Affichage des résultats
        if n_outliers > 0:
            outliers_found = True
            print(f"{col} : {n_outliers} outlier(s) | seuil bas = {lower_bound:.2f} | seuil haut = {upper_bound:.2f}")

    if not outliers_found:
        print("Aucun outlier détecté")


