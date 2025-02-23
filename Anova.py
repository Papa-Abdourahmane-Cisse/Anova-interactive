import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Configuration de la page
st.set_page_config(page_title="Analyse ANOVA Interactive", layout="wide", page_icon="📊")

# Fonction pour charger les données
def load_data():
    uploaded_file = st.file_uploader("📂 Charger le fichier CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=';')  # Utiliser le délimiteur correct
        data.columns = [col.replace(' ', '_').replace('@', '_') for col in data.columns]  # Nettoyer les noms des colonnes
        return data
    return None

# Fonction pour afficher les boxplots
def plot_boxplots(data, x, y):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, y=y, data=data)
    st.pyplot(plt)

# Fonction pour effectuer le test de Shapiro-Wilk
def shapiro_test(data, column):
    stat, p = stats.shapiro(data[column])
    return p

# Fonction pour effectuer le test de Levene
def levene_test(data, column, group):
    groups = data.groupby(group)[column].apply(list)
    if len(groups) < 2:
        raise ValueError("La colonne de groupe doit contenir au moins deux groupes distincts.")
    stat, p = stats.levene(*groups)
    return p

# Fonction pour effectuer l'ANOVA
def anova_analysis(data, formula):
    try:
        model = ols(formula, data=data).fit()
        anova_table = anova_lm(model, typ=2)
        return anova_table
    except Exception as e:
        st.error(f"Erreur lors de l'analyse ANOVA : {e}")
        return None

# Page d'accueil
def home():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #8B4513;">
            <h1 style="color: #004d40; font-size: 2.5em;">📊 Analyse ANOVA Interactive</h1>
            <p style="color: #01579b; font-size: 1.2em;">Bienvenue dans l'application d'analyse de la variance.</p>
            <p style="color: #01579b; font-size: 1.2em;">Cette application vous permet de charger des données, d'effectuer des tests de normalité et d'homoscédasticité, et de réaliser une analyse ANOVA.</p>
            <div style="margin-top: 20px;">
                <a href="#chargement-des-données" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">Charger des données</button>
                </a>
                <a href="#analyse-descriptive" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">Analyse descriptive</button>
                </a>
                <a href="#tests-de-normalité-et-d-homoscédasticité" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">Tests de normalité et d'homoscédasticité</button>
                </a>
                <a href="#anova-sans-interaction" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">ANOVA sans interaction</button>
                </a>
                <a href="#anova-avec-interaction" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">ANOVA avec interaction</button>
                </a>
                <a href="#résultats" style="text-decoration: none;">
                    <button style="background-color: #8B4513; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 1em; margin: 5px;">Résultats</button>
                </a>
            </div>
            <div style="margin-top: 40px;">
                <img src="https://via.placeholder.com/600x400?text=Image+de+fond" alt="Image de fond" style="width: 100%; border-radius: 10px;">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Page de chargement des données
def data_loading():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">📂 Chargement des données</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    data = load_data()
    if data is not None:
        st.write("Aperçu des données :")
        st.dataframe(data.head())
        st.session_state.data = data

# Page d'analyse descriptive
def descriptive_analysis():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">📈 Analyse descriptive</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if 'data' in st.session_state:
        data = st.session_state.data
        st.write("Sélectionnez les colonnes pour les boxplots :")
        x_col = st.selectbox("Variable indépendante", data.columns)
        y_col = st.selectbox("Variable dépendante", data.columns)
        plot_boxplots(data, x_col, y_col)

# Page de tests de normalité et d'homoscédasticité
def normality_homoscedasticity():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">📊 Tests de normalité et d'homoscédasticité</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if 'data' in st.session_state:
        data = st.session_state.data
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        st.write("Sélectionnez la colonne pour le test de Shapiro-Wilk :")
        shapiro_col = st.selectbox("Colonne pour le test de Shapiro-Wilk", numeric_columns)
        p_shapiro = shapiro_test(data, shapiro_col)
        st.write(f"P-valeur du test de Shapiro-Wilk : {p_shapiro}")

        st.write("Sélectionnez les colonnes pour le test de Levene :")
        levene_col = st.selectbox("Colonne pour le test de Levene", numeric_columns)
        group_col = st.selectbox("Groupe pour le test de Levene", data.columns)
        try:
            p_levene = levene_test(data, levene_col, group_col)
            st.write(f"P-valeur du test de Levene : {p_levene}")
        except ValueError as e:
            st.error(e)

# Page d'ANOVA sans interaction
def anova_no_interaction():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">🔍 ANOVA sans interaction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if 'data' in st.session_state:
        data = st.session_state.data
        st.write("Sélectionnez les colonnes pour l'ANOVA sans interaction :")
        y_col = st.selectbox("Variable dépendante", data.columns)
        x_cols = st.multiselect("Variables indépendantes", data.columns)
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        anova_table = anova_analysis(data, formula)
        if anova_table is not None:
            st.write("Tableau ANOVA sans interaction :")
            st.dataframe(anova_table)

# Page d'ANOVA avec interaction
def anova_with_interaction():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">🔍 ANOVA avec interaction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if 'data' in st.session_state:
        data = st.session_state.data
        st.write("Sélectionnez les colonnes pour l'ANOVA avec interaction :")
        y_col = st.selectbox("Variable dépendante", data.columns)
        x_cols = st.multiselect("Variables indépendantes", data.columns)
        interactions = st.multiselect("Interactions", [f"{x1}:{x2}" for x1 in x_cols for x2 in x_cols if x1 != x2])
        formula = f"{y_col} ~ {' + '.join(x_cols)} + {' + '.join(interactions)}"
        anova_table = anova_analysis(data, formula)
        if anova_table is not None:
            st.write("Tableau ANOVA avec interaction :")
            st.dataframe(anova_table)
            st.write("Interprétation des résultats :")
            for index, row in anova_table.iterrows():
                p_value = row['PR(>F)']
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                st.write(f"{index}: {significance}")

# Page de résultats
def results():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; border: 2px solid #8B4513;">
            <h1 style="color: #004d40;">📋 Résultats</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    if 'data' in st.session_state:
        data = st.session_state.data
        st.write("Résultats de l'ANOVA :")
        # Ajouter ici les résultats détaillés

# Menu de navigation
menu = ["Accueil", "Chargement des données", "Analyse descriptive", "Tests de normalité et d'homoscédasticité", "ANOVA sans interaction", "ANOVA avec interaction", "Résultats"]
choice = st.sidebar.selectbox("Menu", menu)

# Ajouter un thème personnalisé avec un arrière-plan bleu clair et afficher le logo
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        background-color: #8B4513;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    .stButton > button:hover {
        background-color: #723F2B;
    }
    .sidebar .sidebar-content {
        background-color: rgba(224, 247, 250, 0.8); /* Bleu clair */
    }
    .sidebar .sidebar-content div:first-child {
        padding-top: 2rem;
    }
    body {
        background-color: #e0f7fa; /* Bleu clair */
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    .logo {
        width: 200px; /* Ajustez la taille du logo selon vos besoins */
    }
    </style>
    """,
    unsafe_allow_html=True
)

if choice == "Accueil":
    home()
elif choice == "Chargement des données":
    data_loading()
elif choice == "Analyse descriptive":
    descriptive_analysis()
elif choice == "Tests de normalité et d'homoscédasticité":
    normality_homoscedasticity()
elif choice == "ANOVA sans interaction":
    anova_no_interaction()
elif choice == "ANOVA avec interaction":
    anova_with_interaction()
elif choice == "Résultats":
    results()
