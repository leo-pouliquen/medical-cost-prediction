import pandas as pd
import numpy as np


# =============================================================================

# Fonction de création de la features obese
def create_feature_obese(df):
    # Indicateur d'obésité basé sur le seuil médical standard BMI >= 30
    df['obese'] = np.where(df['bmi'] >= 30,
                          1,
                          0)
    return df

# =============================================================================

# Fonction de création de la features d'interraction obese_smoker
def create_feature_smoker_obese(df):
    
    # Note : smoker doit être encore en string ('yes'/'no') à ce stade donc avant encode_data
    df['obese_smoker'] = np.where((df['smoker'] == 'yes') & (df['obese'] == 1),
                                                         1,
                                                         0)

    return df

# =============================================================================

# Fonction de création de la features age_group
def create_feature_age_group(df):
    # Segmentation de l'âge en tranches de 5 ans
    # right=False : intervalles fermés à gauche [18, 25[
    df['age_group'] = pd.cut(df['age'],
                            bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
                            labels=["18-25", "26-30", "31-35", "36-40", "41-45",
                                     "46-50", "51-55", "56-60", "61-65"],
                            right=False)
    return df

# =============================================================================

# Fonction chapeau pour executer l'ensemble des créations de features 
def create_features(df):
    df = create_feature_obese(df)
    df = create_feature_smoker_obese(df)
    df = create_feature_age_group(df)

    return df


# =============================================================================

# Fonction de transformation log de la variable charge
def log_charges(df):
    # La colonne charges originale est conservée pour les GLM et modèles ensemblistes
    df['log_charges'] = np.log(df['charges'])

    return df

