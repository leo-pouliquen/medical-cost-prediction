import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fonction de correction des anomalies 
def fix_data (df):
    # On crée un copie de la dataframe pour ne pas modifier l'original
    df =df.copy()

    # Suppression des doublons
    n_before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - df.shape[0]
    print(f"Doublons supprimés : {n_removed} ligne(s) retirée(s)")

    # Calcul des seuils IQR
    Q1 = df["charges"].quantile(0.25)
    Q3 = df["charges"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # Flagging des outliers
    df["is_outlier"] = (df["charges"] < lower_bound) | (df["charges"] > upper_bound)
    n_outliers = df["is_outlier"].sum()
    print(f"Outliers flaggés : {n_outliers} ligne(s) marquée(s) dans 'is_outlier'")

    return df


# =============================================================================

# Fonction d'encodage des variables catégorielles en variables binaires
def encode(df, col, value_for_one, value_for_zero):
    # On crée un copie de la dataframe pour ne pas modifier l'original
    df = df.copy()
    # value_for_one prend 1, value_for_zero prend 0
    df[col] = df[col].map({value_for_one:1, value_for_zero:0})

    return df

# =============================================================================

# Fonction de one hot encode pour les variables avec plusieurs catégories
def one_hot_encode(df, col):
    # drop_first=True supprime une modalité pour éviter la multicolinéarité
    df = pd.get_dummies(df, columns=[col], drop_first= True)

    return df

# =============================================================================

# Fonction chapeau qui exécute l'ensemble des encodages
def encode_data(df):
    df = encode(df, "sex", "female", "male")
    df =encode(df, "smoker", "yes", "no")
    df = one_hot_encode(df, "region")
    df = one_hot_encode(df, 'age_group')
    df = df.drop('age', axis= 1)
    return df

# =============================================================================

# Fonction de split de la base de données 
def split_data(df, col_target, col_stratify):

    # Pour gérer les cas en fonction de la colonne target 
    # On exclut toujours la target et l'autre version de la target (charges ou log_charges)
    # pour éviter toute fuite d'information dans le modèle
    col_to_drop = [col_target]
    if 'charges' in df.columns and col_target != 'charges':
        col_to_drop.append('charges')
    if 'log_charges' in df.columns and col_target != 'log_charges':
        col_to_drop.append('log_charges')

    X = df.drop(col_to_drop, axis = 1)
    y = df[col_target]

    # Split 80/20 stratifié sur smoker pour garantir une proportion identique de fumeurs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=col_stratify)

    print(f"La taille du jeu d'entraînement est de : {X_train.shape}")
    print(f"La taille du jeu de test est de : {X_test.shape}")

    return X_train, X_test, y_train, y_test

# =============================================================================

def build_pipeline(model):
    # Construction d'un pipeline sklearn : StandardScaler + modèle
    # Le pipeline garantit que le scaler est fitté uniquement sur le train set
    # évite tout data leakage lors de l'évaluation sur le test set
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Normalisation 
        ('model', model) # Modèle à entrainer
    ])

    return pipeline

