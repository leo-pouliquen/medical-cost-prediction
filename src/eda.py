import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Style globale appliqué à tous les graphiques du module
sns.set_theme(style="whitegrid")


# =============================================================================
# ANALYSE UNIVARIÉE
# =============================================================================

# Fonction de représentation graphique de la distribution de la target
def plot_target_distributions(df, col_target):
    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize = (16, 5))

    # Distribution normale
    sns.histplot(df[col_target], kde = True, color='royalblue', ax=axes[0])
    axes[0].set_title("Distribution des coûts médicaux")
    axes[0].set_xlabel("Coûts ($)")
    axes[0].set_ylabel("Fréquence")

    # Distribution log
    sns.histplot(np.log(df[col_target]), kde = True, color='darkorange', ax=axes[1])
    axes[1].set_title("Distribution des coûts médicaux (log)")
    axes[1].set_xlabel("Log Coûts ($)")
    axes[1].set_ylabel("Fréquence")


    plt.suptitle("Variable cible : charges", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# =============================================================================

# Fonction de représentation graphique de la distribution des variables numériques
def plot_numerical_features(df, col_num):
    # Grille dynamique adapté au nombre de colonne : pour avoir un graphique par variable
    fig, axes = plt.subplots(nrows = 1, ncols = len(col_num), figsize = (16, 5))

    for i, col in enumerate(col_num):
        # Histogramme
        sns.histplot(df[col], kde = True, color = 'green', ax=axes[i])
        axes[i].set_title(f"Distribution de la feature {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")

    plt.suptitle("Distribution des features numériques", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# =============================================================================

# Fonction de représentation graphique de la distribution des variables catégorielles
def plot_categorial_features(df, col_cat):
    # Grille dynamique adapté au nombre de colonne : pour avoir un graphique par variable
    fig, axes = plt.subplots(nrows = 1, ncols = len(col_cat), figsize = (16, 5))

    for i, col in enumerate(col_cat):
        # Countplot
        sns.countplot(x=df[col], color = 'steelblue', ax=axes[i])
        axes[i].set_title(f"Distribution de la feature {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")

    plt.suptitle("Distribution des features catégorielles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# =============================================================================

# Fonction de résumé des statistiques descriptives
def print_summary_stats(df):
    print("=" * 50)
    print("STATISTIQUES DESCRIPTIVES DU PORTEFEUILLE")
    print("=" * 50)

    # Variables numériques
    print("\nVARIABLES NUMÉRIQUES")
    print("=" * 30)
    print(f"Age moyen        : {df['age'].mean():.1f} ans")
    print(f"Age médian       : {df['age'].median():.1f} ans")
    print(f"Age min / max    : {df['age'].min()} / {df['age'].max()} ans")
    print(f"BMI moyen        : {df['bmi'].mean():.1f}")
    print(f"Nb enfants moyen : {df['children'].mean():.1f}")

    # Variables catégorielles
    print("\nVARIABLES CATÉGORIELLES")
    print("=" * 30)
    pct_smoker = df["smoker"].value_counts(normalize=True)["yes"] * 100
    pct_female = df["sex"].value_counts(normalize=True)["female"] * 100
    print(f"Proportion fumeurs  : {pct_smoker:.1f}%")
    print(f"Proportion femmes   : {pct_female:.1f}%")
    print("\nRépartition par région :")
    print((df["region"].value_counts(normalize=True) * 100).round(1).to_string())

    # Variable cible
    print("\nVARIABLE CIBLE : CHARGES")
    print("=" * 30)
    print(f"Coût moyen global         : {df['charges'].mean():,.0f}$")
    print(f"Coût médian global        : {df['charges'].median():,.0f}$")
    print(f"\nCoût moyen fumeurs        : {df[df['smoker'] == 'yes']['charges'].mean():,.0f}$")
    print(f"Coût moyen non-fumeurs    : {df[df['smoker'] == 'no']['charges'].mean():,.0f}$")
    print(f"\nCoût moyen par région :")
    print(df.groupby("region")["charges"].mean().apply(lambda x: f"{x:,.0f}$").to_string())

    print("\n" + "=" * 50)   


# =============================================================================
# ANALYSE BIVARIÉE
# =============================================================================

# Fonction de représentation graphique de la target vs les variables numériques
def plot_target_vs_num(df, col_target, col_num):
    fig, axes = plt.subplots(nrows = 1, ncols = len(col_num), figsize=(16,5))

    for i, col in enumerate(col_num):
        # Regplot
        # alpha=0.3 pour mieux visualiser les zones de densité
        sns.regplot(data=df, x=col, y=col_target, ax=axes[i], scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        axes[i].set_title(f"Relation entre la variable {col} et {col_target}")

    plt.tight_layout()
    plt.show()

# =============================================================================

# Fonction de représentation graphique de la target vs les variables catégorielles
def plot_target_vs_cat(df, col_target, col_cat):
    fig, axes = plt.subplots(nrows=1, ncols=len(col_cat), figsize=(16,5))

    for i, col in enumerate(col_cat):
        # Boxplot avec hue = smoker (révèle l'effet fumeur pour les autres variables)
        sns.boxplot(data=df, x=col, y=col_target, ax=axes[i], palette='viridis', hue='smoker')
        axes[i].set_title(f"Relation entre la variable {col} et {col_target}")

    plt.tight_layout()
    plt.show()

# =============================================================================

# Fonction de représentation graphique des coûts moyens en fonction des variables catégorielles
def plot_means_charges_by_cat(df, col_target, col_cat):
        fig, axes = plt.subplots(1, len(col_cat), figsize=(16, 5))

        for i, col in enumerate(col_cat):

            # Calcul du coût moyen par catégorie
            mean_charges = df.groupby(col)[col_target].mean().reset_index()

            # Barplot
            sns.barplot(data= mean_charges, x=col, y=col_target, palette='viridis', ax=axes[i], hue=col, legend= False)
            axes[i].set_title(f"Coût moyen par {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Coût moyen ($)")


            # Afficher les valeurs au dessus de chaque barre
            for bar in axes[i].patches:
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 100,
                    f"{bar.get_height():,.0f}$",
                    ha="center", va="bottom", fontsize=10
                )

        plt.suptitle("Coût moyen par segment", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

# =============================================================================

# Tests statistiques 
def run_statistical_tests(df, col_target, col_num, col_cat):

    # Création d'une liste de stockage des résultats
    results = []

    # Corrélation de Pearson pour les variables numériques
    for col in col_num:
        stat, pvalue = stats.pearsonr(df[col], df[col_target])
        results.append({
            'variable': col,
            'test':"Pearson", 
            'statistique':round(stat, 4),
            'p_value':round(pvalue, 4),
            'significatif':'Oui' if pvalue < 0.05 else 'Non'
        })

    # ANOVA pour les variables catégorielles
    for col in col_cat:
        groups = [group[col_target].values for _, group in df.groupby(col)]
        stat, pvalue = stats.f_oneway(*groups)
        results.append({
            "variable": col,
            "test": "ANOVA",
            "statistique": round(stat, 4),
            "p_value": round(pvalue, 4),
            "significatif": "Oui" if pvalue < 0.05 else "Non"
        })

    print("\nTESTS STATISTIQUES — RELATION AVEC", col_target.upper())
    print("-" * 60)
    print(pd.DataFrame(results).to_string(index=False))
    print("-" * 60)
    print("Seuil de significativité : p-value < 0.05")


# =============================================================================
# ANALYSE MULTIVARIÉE
# =============================================================================

# Matrice de corrélation
def plot_correlation_matrix(df):

    # Encodage 
    cols = ['bmi', 'children', 'charges', 'log_charges', 'smoker', 'sex', 'obese', 'obese_smoker']
    corr = df[cols].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap = 'coolwarm', annot= True)
    plt.title('Matrice de corrélation')
    plt.show()

# =============================================================================

# Pairplot coloré par statut fumeur
def plot_pairplot(df):
    
    # rouge = fumeurs, bleu = non-fumeurs
    sns.pairplot(df, hue="smoker", palette={"yes": "red", "no": "steelblue"},plot_kws={"alpha": 0.3})
    plt.suptitle("Pairplot en fonction du statut fumeur", fontsize=14, fontweight="bold", y=1.02)
    plt.show()






