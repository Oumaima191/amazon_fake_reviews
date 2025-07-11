import pandas as pd
import numpy as np
import re
import string
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Télécharger les stopwords une seule fois
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#  1. Charger les données issues du scraping
df = pd.read_csv("amazon_avis_textes.csv")

#  2. Nettoyage basique
df = df.drop_duplicates()
df = df.dropna(subset=["texte_avis"])

# Supprimer les avis trop courts (moins de 10 caractères par ex.)
df = df[df["texte_avis"].str.len() > 10]

#  3. Fonction de nettoyage de texte
def clean_text(text):
    text = str(text)
    text = unidecode.unidecode(text)                         # enlever les accents
    text = text.lower()                                      # tout en minuscules
    text = re.sub(r'\d+', '', text)                          # supprimer les chiffres
    text = text.translate(str.maketrans('', '', string.punctuation))  # supprimer la ponctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]        # enlever les stopwords
    return " ".join(words)

#  4. Nettoyer les avis
df["texte_nettoye"] = df["texte_avis"].apply(clean_text)

#  5. Générer des labels aléatoires (0 = vrai, 1 = faux) — temporairement
df["label"] = np.random.randint(0, 2, size=len(df))

# 6. Sauvegarder les résultats
df.to_csv("avis_labellises.csv", index=False)

# 7. Affichage pour contrôle
print(df[["texte_avis", "texte_nettoye", "label"]].head())
print(f"\n Nettoyage et labellisation terminés : {len(df)} lignes enregistrées dans 'avis_labellises.csv'")
