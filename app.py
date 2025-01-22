import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score

# Titre de l'application
st.title("Prédiction de Puissance (MW) ")
st.write("Cette application utilise un modèle KNN pour prédire la puissance (MW) en fonction de diverses caractéristiques.")

# Section pour charger le dataset
st.sidebar.header("Charger un dataset")
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier CSV", type=["csv"])

if uploaded_file:
    # Lecture du fichier CSV
    df = pd.read_csv(uploaded_file)

    # Aperçu des données
    st.write("Aperçu du dataset :")
    st.dataframe(df.head())

    # Sélection des colonnes utiles
    if 'Puissance (MW)' in df.columns:
        X = df.drop(columns=['Puissance (MW)'])
        y = df['Puissance (MW)']

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

        # Modèle KNN
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(X_train, y_train)

        # Performances du modèle
        y_pred = knn.predict(X_test)
        st.write("Performances du modèle :")
        st.write(f"Score R² moyen : {r2_score(y_test, y_pred):.3f}")
        st.write(f"Explained Variance Score : {explained_variance_score(y_test, y_pred):.3f}")
        st.write(f"Mean Absolute Error : {mean_absolute_error(y_test, y_pred):.3f}")

        # Prédictions utilisateur
        st.sidebar.header("Faire une prédiction")
        st.sidebar.write("Entrez les valeurs pour prédire la puissance (MW) :")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.sidebar.number_input(f"{col}", value=0.0, format="%.2f")

        # Conversion des inputs en DataFrame
        input_df = pd.DataFrame([input_data])

        if st.sidebar.button("Prédire"):
            prediction = knn.predict(input_df)[0]
            st.sidebar.write(f"### La puissance prédite : {prediction:.2f} MW")
    else:
        st.error("Le fichier doit contenir une colonne nommée 'Puissance (MW)'.")
else:
    st.info("Veuillez télécharger un dataset au format CSV.")
