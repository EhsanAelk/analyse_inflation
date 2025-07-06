import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Impact Politiques Publiques - Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("Analyse de l'Impact des Politiques Publiques sur l'Emploi et l'Inflation au Maroc")
st.markdown("---")

# Fonction pour générer des données fictives réalistes
@st.cache_data
def generate_moroccan_data():
    """Génère des données fictives réalistes pour le Maroc"""
    np.random.seed(42)
    
    # Période d'analyse : 2015-2024
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')
    n_periods = len(dates)
    
    # Génération des données macro-économiques
    base_inflation = 1.5  # Inflation de base au Maroc
    base_unemployment = 9.5  # Taux de chômage de base
    base_gdp_growth = 3.2  # Croissance PIB de base
    
    # Simulation des politiques publiques (variables binaires)
    politique_emploi = np.random.choice([0, 1], n_periods, p=[0.7, 0.3])  # Programme emploi jeunes
    politique_fiscale = np.random.choice([0, 1], n_periods, p=[0.8, 0.2])  # Réformes fiscales
    politique_monetaire = np.random.choice([0, 1], n_periods, p=[0.6, 0.4])  # Politique monétaire expansive
    investissement_public = np.random.choice([0, 1], n_periods, p=[0.5, 0.5])  # Investissement infrastructure
    
    # Génération des séries temporelles avec tendances et saisonnalité
    trend = np.linspace(0, 2, n_periods)
    seasonal = np.sin(2 * np.pi * np.arange(n_periods) / 12)
    
    # Taux d'inflation (influencé par les politiques)
    inflation_noise = np.random.normal(0, 0.5, n_periods)
    inflation = (base_inflation + 
                0.3 * trend + 
                0.2 * seasonal + 
                -0.8 * politique_monetaire + 
                0.5 * investissement_public + 
                inflation_noise)
    inflation = np.clip(inflation, -1, 8)  # Limiter les valeurs extrêmes
    
    # Taux de chômage (influencé par les politiques)
    unemployment_noise = np.random.normal(0, 0.3, n_periods)
    unemployment = (base_unemployment + 
                   -0.2 * trend + 
                   0.1 * seasonal + 
                   -1.2 * politique_emploi + 
                   -0.7 * investissement_public + 
                   unemployment_noise)
    unemployment = np.clip(unemployment, 5, 15)  # Limiter les valeurs réalistes
    
    # Croissance du PIB
    gdp_noise = np.random.normal(0, 0.8, n_periods)
    gdp_growth = (base_gdp_growth + 
                  0.1 * trend + 
                  0.3 * seasonal + 
                  0.9 * politique_emploi + 
                  1.2 * investissement_public + 
                  gdp_noise)
    gdp_growth = np.clip(gdp_growth, -2, 8)
    
    # Autres indicateurs économiques
    prix_petrole = 60 + 20 * np.sin(2 * np.pi * np.arange(n_periods) / 24) + np.random.normal(0, 5, n_periods)
    taux_change = 9.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_periods) / 36) + np.random.normal(0, 0.2, n_periods)
    indice_confiance = 50 + 10 * np.sin(2 * np.pi * np.arange(n_periods) / 18) + np.random.normal(0, 3, n_periods)
    
    # Création du DataFrame
    data = pd.DataFrame({
        'date': dates,
        'inflation': inflation,
        'chomage': unemployment,
        'croissance_pib': gdp_growth,
        'politique_emploi': politique_emploi,
        'politique_fiscale': politique_fiscale,
        'politique_monetaire': politique_monetaire,
        'investissement_public': investissement_public,
        'prix_petrole': prix_petrole,
        'taux_change': taux_change,
        'indice_confiance': indice_confiance
    })
    
    return data

# Fonction pour l'analyse de corrélation
def analyze_correlations(data):
    """Analyse les corrélations entre variables"""
    correlation_vars = ['inflation', 'chomage', 'croissance_pib', 'politique_emploi', 
                       'politique_fiscale', 'politique_monetaire', 'investissement_public']
    corr_matrix = data[correlation_vars].corr()
    return corr_matrix

# Fonction pour les modèles de machine learning
def build_ml_models(data):
    """Construit des modèles ML pour prédire l'inflation et le chômage"""
    features = ['politique_emploi', 'politique_fiscale', 'politique_monetaire', 
               'investissement_public', 'prix_petrole', 'taux_change', 'indice_confiance']
    
    # Modèle pour l'inflation
    X_inflation = data[features]
    y_inflation = data['inflation']
    X_train_inf, X_test_inf, y_train_inf, y_test_inf = train_test_split(
        X_inflation, y_inflation, test_size=0.2, random_state=42
    )
    
    rf_inflation = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_inflation.fit(X_train_inf, y_train_inf)
    y_pred_inf = rf_inflation.predict(X_test_inf)
    r2_inf = r2_score(y_test_inf, y_pred_inf)
    
    # Modèle pour le chômage
    X_chomage = data[features]
    y_chomage = data['chomage']
    X_train_cho, X_test_cho, y_train_cho, y_test_cho = train_test_split(
        X_chomage, y_chomage, test_size=0.2, random_state=42
    )
    
    rf_chomage = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_chomage.fit(X_train_cho, y_train_cho)
    y_pred_cho = rf_chomage.predict(X_test_cho)
    r2_cho = r2_score(y_test_cho, y_pred_cho)
    
    return {
        'inflation_model': rf_inflation,
        'chomage_model': rf_chomage,
        'inflation_r2': r2_inf,
        'chomage_r2': r2_cho,
        'features': features
    }

# Chargement des données
data = generate_moroccan_data()

# Sidebar pour les options
st.sidebar.header("🔧 Options d'Analyse")
analysis_type = st.sidebar.selectbox(
    "Type d'analyse",
    ["Vue d'ensemble", "Analyse temporelle", "Impact des politiques", "Modèles prédictifs", "Scénarios"]
)

# Section principale
if analysis_type == "Vue d'ensemble":
    st.header("📊 Vue d'Ensemble des Données")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inflation Moyenne", f"{data['inflation'].mean():.2f}%")
    with col2:
        st.metric("Chômage Moyen", f"{data['chomage'].mean():.2f}%")
    with col3:
        st.metric("Croissance PIB Moyenne", f"{data['croissance_pib'].mean():.2f}%")
    with col4:
        st.metric("Période d'Analyse", f"{len(data)} mois")
    
    st.subheader("📈 Évolution des Indicateurs Principaux")
    
    # Graphique des séries temporelles
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Inflation (%)', 'Taux de Chômage (%)', 'Croissance PIB (%)', 'Politiques Publiques'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(x=data['date'], y=data['inflation'], name='Inflation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['chomage'], name='Chômage'), row=1, col=2)
    fig.add_trace(go.Scatter(x=data['date'], y=data['croissance_pib'], name='PIB'), row=2, col=1)
    
    # Graphique des politiques
    fig.add_trace(go.Scatter(x=data['date'], y=data['politique_emploi'], name='Politique Emploi'), row=2, col=2)
    fig.add_trace(go.Scatter(x=data['date'], y=data['investissement_public'], name='Invest. Public'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔍 Statistiques Descriptives")
    st.dataframe(data.describe())

elif analysis_type == "Analyse temporelle":
    st.header("📅 Analyse Temporelle")
    
    # Sélection de la variable à analyser
    variable = st.selectbox(
        "Variable à analyser",
        ["inflation", "chomage", "croissance_pib"]
    )
    
    # Graphique temporel détaillé
    fig = px.line(data, x='date', y=variable, 
                  title=f'Évolution de {variable.replace("_", " ").title()}')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de saisonnalité
    st.subheader("🔄 Analyse de Saisonnalité")
    data_monthly = data.copy()
    data_monthly['mois'] = data_monthly['date'].dt.month
    monthly_avg = data_monthly.groupby('mois')[variable].mean()
    
    fig_seasonal = px.bar(x=monthly_avg.index, y=monthly_avg.values,
                         title=f'Moyenne mensuelle - {variable.replace("_", " ").title()}')
    fig_seasonal.update_layout(height=400)
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Tendance et décomposition
    st.subheader("📈 Analyse de Tendance")
    window = st.slider("Fenêtre de lissage (mois)", 3, 24, 12)
    
    data_trend = data.copy()
    data_trend['tendance'] = data_trend[variable].rolling(window=window).mean()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=data_trend['date'], y=data_trend[variable], 
                                  name='Valeur originale', opacity=0.7))
    fig_trend.add_trace(go.Scatter(x=data_trend['date'], y=data_trend['tendance'], 
                                  name='Tendance', line=dict(width=3)))
    fig_trend.update_layout(title='Tendance vs Valeurs Originales', height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

elif analysis_type == "Impact des politiques":
    st.header("🏛️ Impact des Politiques Publiques")
    
    # Analyse comparative avant/après politiques
    st.subheader("📊 Comparaison Avant/Après Politiques")
    
    politique_select = st.selectbox(
        "Politique à analyser",
        ["politique_emploi", "politique_fiscale", "politique_monetaire", "investissement_public"]
    )
    
    # Données avec et sans politique
    with_policy = data[data[politique_select] == 1]
    without_policy = data[data[politique_select] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Inflation (Avec politique)", f"{with_policy['inflation'].mean():.2f}%")
        st.metric("Chômage (Avec politique)", f"{with_policy['chomage'].mean():.2f}%")
    
    with col2:
        st.metric("Inflation (Sans politique)", f"{without_policy['inflation'].mean():.2f}%")
        st.metric("Chômage (Sans politique)", f"{without_policy['chomage'].mean():.2f}%")
    
    # Graphique de comparaison
    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=('Inflation', 'Chômage'))
    
    fig_comp.add_trace(go.Box(y=with_policy['inflation'], name='Avec Politique'), row=1, col=1)
    fig_comp.add_trace(go.Box(y=without_policy['inflation'], name='Sans Politique'), row=1, col=1)
    
    fig_comp.add_trace(go.Box(y=with_policy['chomage'], name='Avec Politique'), row=1, col=2)
    fig_comp.add_trace(go.Box(y=without_policy['chomage'], name='Sans Politique'), row=1, col=2)
    
    fig_comp.update_layout(height=400)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Matrice de corrélation
    st.subheader("🔗 Matrice de Corrélation")
    corr_matrix = analyze_correlations(data)
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Corrélations entre Variables")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Modèles prédictifs":
    st.header("🤖 Modèles Prédictifs")
    
    # Construction des modèles
    models = build_ml_models(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Performance Modèle Inflation (R²)", f"{models['inflation_r2']:.3f}")
    with col2:
        st.metric("Performance Modèle Chômage (R²)", f"{models['chomage_r2']:.3f}")
    
    # Importance des variables
    st.subheader("📊 Importance des Variables")
    
    # Inflation
    inf_importance = pd.DataFrame({
        'Variable': models['features'],
        'Importance': models['inflation_model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp_inf = px.bar(inf_importance, x='Importance', y='Variable', 
                        title='Importance des Variables - Modèle Inflation')
    fig_imp_inf.update_layout(height=400)
    st.plotly_chart(fig_imp_inf, use_container_width=True)
    
    # Chômage
    cho_importance = pd.DataFrame({
        'Variable': models['features'],
        'Importance': models['chomage_model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp_cho = px.bar(cho_importance, x='Importance', y='Variable', 
                        title='Importance des Variables - Modèle Chômage')
    fig_imp_cho.update_layout(height=400)
    st.plotly_chart(fig_imp_cho, use_container_width=True)
    
    # Prédictions
    st.subheader("🔮 Prédictions")
    
    # Interface pour faire des prédictions
    st.write("Configurez un scénario pour obtenir des prédictions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pol_emploi = st.selectbox("Politique Emploi", [0, 1])
        pol_fiscale = st.selectbox("Politique Fiscale", [0, 1])
        pol_monetaire = st.selectbox("Politique Monétaire", [0, 1])
        invest_public = st.selectbox("Investissement Public", [0, 1])
    
    with col2:
        prix_petrole = st.slider("Prix du Pétrole", 30, 120, 70)
        taux_change = st.slider("Taux de Change", 8.0, 12.0, 10.0)
        indice_confiance = st.slider("Indice de Confiance", 20, 80, 50)
    
    # Prédiction
    scenario = np.array([[pol_emploi, pol_fiscale, pol_monetaire, invest_public, 
                         prix_petrole, taux_change, indice_confiance]])
    
    pred_inflation = models['inflation_model'].predict(scenario)[0]
    pred_chomage = models['chomage_model'].predict(scenario)[0]
    
    st.write("### Prédictions pour ce scénario:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Inflation Prédite", f"{pred_inflation:.2f}%")
    with col2:
        st.metric("Chômage Prédit", f"{pred_chomage:.2f}%")

elif analysis_type == "Scénarios":
    st.header("🎯 Analyse de Scénarios")
    
    st.write("Analysez l'impact de différents scénarios de politiques publiques:")
    
    # Définition des scénarios
    scenarios = {
        "Scénario de Base": [0, 0, 0, 0],
        "Politique Pro-Emploi": [1, 0, 0, 1],
        "Politique Monétaire Expansive": [0, 1, 1, 0],
        "Politique Complète": [1, 1, 1, 1]
    }
    
    # Paramètres économiques fixes
    prix_petrole = 70
    taux_change = 10.0
    indice_confiance = 50
    
    # Calcul des prédictions pour chaque scénario
    models = build_ml_models(data)
    results = []
    
    for scenario_name, policies in scenarios.items():
        scenario_data = np.array([policies + [prix_petrole, taux_change, indice_confiance]])
        pred_inf = models['inflation_model'].predict(scenario_data)[0]
        pred_cho = models['chomage_model'].predict(scenario_data)[0]
        
        results.append({
            'Scénario': scenario_name,
            'Inflation': pred_inf,
            'Chômage': pred_cho,
            'Politique Emploi': policies[0],
            'Politique Fiscale': policies[1],
            'Politique Monétaire': policies[2],
            'Investissement Public': policies[3]
        })
    
    results_df = pd.DataFrame(results)
    
    # Affichage des résultats
    st.subheader("📊 Résultats des Scénarios")
    st.dataframe(results_df)
    
    # Graphique de comparaison
    fig_scenarios = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Impact sur l\'Inflation', 'Impact sur le Chômage')
    )
    
    fig_scenarios.add_trace(
        go.Bar(x=results_df['Scénario'], y=results_df['Inflation'], name='Inflation'),
        row=1, col=1
    )
    
    fig_scenarios.add_trace(
        go.Bar(x=results_df['Scénario'], y=results_df['Chômage'], name='Chômage'),
        row=1, col=2
    )
    
    fig_scenarios.update_layout(height=500)
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Analyse coût-bénéfice simplifiée
    st.subheader("💰 Analyse Coût-Bénéfice")
    
    # Calcul d'un score composite (plus le chômage est bas et l'inflation stable, mieux c'est)
    results_df['Score'] = (10 - results_df['Chômage']) - abs(results_df['Inflation'] - 2)
    
    fig_score = px.bar(results_df, x='Scénario', y='Score', 
                      title='Score Composite des Scénarios')
    fig_score.update_layout(height=400)
    st.plotly_chart(fig_score, use_container_width=True)
    
    st.write("**Score Composite**: Plus le score est élevé, meilleur est le scénario (basé sur un chômage faible et une inflation proche de 2%)")
