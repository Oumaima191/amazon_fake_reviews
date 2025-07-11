import seaborn as sns
import matplotlib.pyplot as plt

# 1. Calculer la longueur des avis nettoyés
df["longueur_avis"] = df["texte_nettoye"].apply(len)

# 2. Score de suspicion basé sur des règles simples
def suspicion_score(row):
    score = 0

    # Critère 1 : avis très court ou très long
    if row["longueur_avis"] < 50 or row["longueur_avis"] > 1000:
        score += 1

    # Critère 2 : contenu trop générique (ex: mots typiques des faux avis)
    generic_terms = ["great", "perfect", "excellent", "amazing", "love", "good quality", "best product"]
    if any(term in row["texte_nettoye"] for term in generic_terms):
        score += 1

    # Critère 3 : si label = 1 (faux dans ce cas, simulation)
    if row["label"] == 1:
        score += 1

    return score

df["suspicion_score"] = df.apply(suspicion_score, axis=1)

# Distribution des longueurs d’avis
sns.histplot(df["longueur_avis"], bins=30, color="skyblue")
plt.title("Distribution des longueurs d'avis")
plt.xlabel("Longueur du texte")
plt.ylabel("Nombre d'avis")
plt.show()

#Distribution du score de suspicion
sns.countplot(x="suspicion_score", data=df, palette="Set2")
plt.title("Distribution des scores de suspicion")
plt.xlabel("Score de suspicion")
plt.ylabel("Nombre d'avis")
plt.show()

#Tableau final des avis suspects
# Tri des avis suspects
df_suspects = df.sort_values(by="suspicion_score", ascending=False)

# Afficher les 10 plus suspects
print(df_suspects[["texte_avis", "texte_nettoye", "suspicion_score", "label"]].head(10))

