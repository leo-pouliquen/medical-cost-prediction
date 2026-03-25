import pandas as pd
import os
import sys

# Ajout de la racine du projet au path pour trouver config.py depuis src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fonction de chargement des données 
def data_loading(file_path):
    # Vérification que le fichier existe avant de tenter le chargement
    # Message d'erreur si le fichier est introuvable
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    
    # Chargement du CSV en dataframe
    df = pd.read_csv(file_path)

    # Confirmation du téléchargement et affichage des dimensions du dataset
    print(f"Téléchargement réussi - {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


