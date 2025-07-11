import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Charger le fichier du scraping avec avis
df = pd.read_csv("amazon_avis_textes.csv")

# Nettoyage basique
df = df.dropna(subset=["texte_avis"])
df = df[df["texte_avis"].str.len() > 10]

# Nettoyage et conversion de colonnes numériques

# Prix (enlever $ et convertir)
df['prix_nettoye'] = df['prix'].str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '').astype(float, errors='ignore')

# Note (ex : '4.5 out of 5 stars' -> 4.5)
df['note_num'] = df['note'].str.extract(r'([\d.]+)').astype(float)

# Nombre d'avis
df['nb_avis_num'] = pd.to_numeric(df['nb_avis'].str.replace(',', ''), errors='coerce')

#  Statistiques descriptives
print("\nStatistiques descriptives des prix :")
print(df['prix_nettoye'].describe())

print("\nStatistiques descriptives des notes :")
print(df['note_num'].describe())

print("\nStatistiques descriptives du nombre d'avis :")
print(df['nb_avis_num'].describe())

#  Longueur des textes d’avis
df['longueur_texte'] = df['texte_avis'].apply(lambda x: len(str(x).split()))

print("\n📏 Statistiques des longueurs des avis :")
print(df['longueur_texte'].describe())

#  Visualisations

# Histogramme des prix
plt.figure(figsize=(8,5))
sns.histplot(df['prix_nettoye'].dropna(), bins=30, kde=True)
plt.title("Distribution des prix")
plt.xlabel("Prix (USD)")
plt.ylabel("Nombre de produits")
plt.show()

# Histogramme des notes
plt.figure(figsize=(8,5))
sns.countplot(x='note_num', data=df)
plt.title("Distribution des notes")
plt.xlabel("Note")
plt.ylabel("Nombre de produits")
plt.show()

# Histogramme du nombre d’avis
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(df['nb_avis_num'].dropna()), bins=30, kde=True)
plt.title("Distribution logarithmique du nombre d'avis")
plt.xlabel("Log(1 + nombre d'avis)")
plt.ylabel("Nombre de produits")
plt.show()

# Longueur des avis
plt.figure(figsize=(8,5))
sns.histplot(df['longueur_texte'], bins=30, kde=True)
plt.title("Distribution de la longueur des avis")
plt.xlabel("Nombre de mots dans l’avis")
plt.ylabel("Nombre de produits")
plt.show()

# Relation prix vs note
plt.figure(figsize=(8,5))
sns.scatterplot(x='prix_nettoye', y='note_num', data=df)
plt.title("Note en fonction du prix")
plt.xlabel("Prix (USD)")
plt.ylabel("Note")
plt.show()

# Relation note vs nombre d'avis
plt.figure(figsize=(8,5))
sns.boxplot(x='note_num', y='nb_avis_num', data=df)
plt.title("Nombre d'avis en fonction de la note")
plt.xlabel("Note")
plt.ylabel("Nombre d'avis")
plt.yscale('log')
plt.show()
