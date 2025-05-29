import pandas as pd
import pickle as pk
import streamlit as st

# Correction ici : bonne syntaxe de open() + load()
model = pk.load(open(r"C:\Users\MADE\Desktop\Data Science\PredictionDesPrixAppartements\prediction_Prix_Appartements.pkl", "rb"))

st.header("DomoPred: Prédicteur de prix de maison")

# Chargement du dataset
data = pd.read_csv(r"C:\Users\MADE\Desktop\Data Science\PredictionDesPrixAppartements\Dataset_Propre.csv")

# Saisie utilisateur
localite = st.selectbox("Veuillez choisir la localité: ", data['location'].unique())
qsft = st.number_input("Entrez la superficie totale souhaitée:")
salle_bain = st.number_input("Entrez le nombre de salle(s) de bain: ")
balcons = st.number_input("Entrez le nombre de balcon(s):")
chambre = st.number_input("Entrez le nombre de chambre(s):")

# Correction ici : création correcte du DataFrame avec les vraies variables
entree = pd.DataFrame([[localite, qsft, salle_bain, balcons, chambre, ]],
                      columns=['location', 'total_sqft', 'bath','balcony', 'bedrooms', ])

# Prédiction
if st.button("Prédire le prix"):
    output = model.predict(entree)
    out_str = "Le prix de l'appartement est : " + str(round(output[0] * 1000, 2)) + " FCFA"
    st.success(out_str)