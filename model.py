# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Charger le dataset labellisé et nettoyé
# df = pd.read_csv("avis_labellises.csv")

# # Séparer les features (texte nettoyé) et la cible (label)
# X = df["titre_nettoye"]
# y = df["label"]

# # Diviser en train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Vectorisation TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Créer et entraîner le modèle de régression logistique
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_tfidf, y_train)

# # Prédictions sur le test
# y_pred = model.predict(X_test_tfidf)

# # Évaluation
# acc = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc:.2f}")

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Matrice de confusion
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=["Vrai", "Faux"], yticklabels=["Vrai", "Faux"])
# plt.xlabel("Prédit")
# plt.ylabel("Réel")
# plt.title("Matrice de confusion")
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  Charger les données labellisées
df = pd.read_csv("avis_labellises.csv")

#  Variables explicative (X) et cible (y)
X = df["texte_nettoye"]  # <-- on utilise maintenant le texte nettoyé
y = df["label"]

#  Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#  Définir les modèles à tester
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

#  Entraînement & Évaluation
for name, model in models.items():
    print(f"\n--- Modèle : {name} ---")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Vrai", "Faux"], yticklabels=["Vrai", "Faux"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {name}")
    plt.show()

