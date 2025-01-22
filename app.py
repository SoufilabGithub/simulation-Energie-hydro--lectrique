import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
from datetime import timedelta
import streamlit as st

# 1. Importation des données
st.title("Analyse et Prévision de la Puissance Hydroélectrique")
st.write("Cette application prédit la puissance hydroélectrique basée sur un modèle de régression KNN.")

uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type="xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file, index_col='date', parse_dates=True)
    st.write("### Aperçu des données chargées :")
    st.dataframe(df.head())

    # Analyse exploratoire
    st.write("### Analyse Exploratoire des Données")
    st.write(f"**Dimensions** : {df.shape}")
    st.write(f"**Nombre d'éléments** : {df.size}")
    st.write("**Types des données** :")
    st.write(df.dtypes)
    st.write("**Statistiques descriptives** :")
    st.dataframe(df.describe())
    
    # Matrice de corrélation
    st.write("**Corrélation entre les variables** :")
    corr = df.corr()
    st.dataframe(corr.style.background_gradient(cmap='Greens'))

    # Données manquantes
    st.write("**Données manquantes** :")
    st.write(df.isnull().sum())

    # Nettoyage des données
    df = df.dropna()

    # 3. Modélisation et prédictions
    st.write("### Modélisation et Prédictions")
    X = df.drop(columns=['puissance (kwatts)'], errors='ignore')
    y = df['puissance (kwatts)']

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

    # Entraîner le modèle
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)

    # Prédictions et performances
    y_pred = knn.predict(X_test)
    st.write(f"**Score R²** : {r2_score(y_test, y_pred):.3f}")
    st.write(f"**Explained Variance Score** : {explained_variance_score(y_test, y_pred):.3f}")
    st.write(f"**Mean Absolute Error** : {mean_absolute_error(y_test, y_pred):.3f}")

    # Prédictions utilisateur
    st.write("### Prédiction Personnalisée")
    input_values = {}
    for col in X.columns:
        input_values[col] = st.number_input(f"Entrez la valeur pour '{col}'", value=0.0)

    if st.button("Prédire"):
        input_df = pd.DataFrame([input_values])
        prediction = knn.predict(input_df)[0]
        st.write(f"### La puissance prédite est : {prediction:.2f} kW")

    # 4. Prévisions temporelles
    st.write("### Prévisions Temporelles")
    n = st.number_input("Entrez le nombre de mois à prévoir", min_value=1, step=1, value=1)

    if st.button("Prévoir pour les périodes futures"):
        last_date = df.index[-1]
        new_dates = [last_date + timedelta(days=30 * i) for i in range(1, n + 1)]
        new_data = pd.DataFrame({'date': new_dates})

        for i in range(n):
            prediction = knn.predict([X_test.iloc[0]])  # Exemple basé sur les données existantes
            new_data.loc[i, 'puissance (kwatts)'] = prediction[0]

        st.write("### Prévisions pour les périodes futures :")
        st.dataframe(new_data)
