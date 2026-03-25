import shap
import matplotlib.pyplot as plt

# =============================================================================

def shap_summary_plot(pipeline, X_test):
    # Extraction du modèle depuis le pipeline
    model = pipeline.named_steps["model"]

    # Calcul des SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Importance globale des variables (SHAP)")
    plt.tight_layout()
    plt.show()
    

# =============================================================================

def shap_waterfall_plot(pipeline, X_test, index=0):
    
    # Extraction du modèle depuis le pipeline
    model = pipeline.named_steps["model"]
    
    # Calcul des SHAP values avec l'API objet — nécessaire pour le waterfall plo
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Waterfall plot pour un assuré individuel
    # Barres rouges : variables qui augmentent le coût prédit
    # Barres bleues : variables qui réduisent le coût prédit
    shap.plots.waterfall(shap_values[index], show=False)
    plt.gcf().set_size_inches(12, 8)
    plt.gcf().suptitle(f"Explication individuelle — assuré n°{index}", fontsize=14, fontweight="bold")    
    plt.tight_layout()
    plt.show()