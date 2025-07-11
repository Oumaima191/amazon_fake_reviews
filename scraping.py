import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
}

NB_PAGES = 10  # Commence avec 2 ou 3 pour tester (tu peux l’augmenter ensuite)
BASE_URL = "https://www.amazon.com/s?k=wireless+headphones&i=electronics&page="

titles, prices, ratings, reviews, review_texts = [], [], [], [], []

for page in range(1, NB_PAGES + 1):
    print(f"\n Scraping page {page}...")
    url = BASE_URL + str(page)
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    product_blocks = soup.select("div.s-main-slot div.s-result-item")

    for product in product_blocks:
        try:
            # Titre
            title = product.h2.get_text(strip=True) if product.h2 else ""

            # Prix
            price_tag = product.select_one("span.a-price > span.a-offscreen")
            price = price_tag.get_text(strip=True) if price_tag else ""

            # Note
            rating_tag = product.select_one("span.a-icon-alt")
            rating = rating_tag.get_text(strip=True) if rating_tag else ""

            # Nombre d’avis
            review_count_tag = product.select_one("span.a-size-base")
            review_count = review_count_tag.get_text(strip=True) if review_count_tag else ""

            # Lien vers la page produit
            link_tag = product.select_one("a.a-link-normal.s-no-outline")
            if not link_tag:
                continue

            raw_href = link_tag.get("href")
            if raw_href.startswith("http"):
                product_link = raw_href
            else:
                product_link = "https://www.amazon.com" + raw_href

            # Scraper la page produit pour récupérer le premier avis
            product_resp = requests.get(product_link, headers=HEADERS)
            product_soup = BeautifulSoup(product_resp.content, "html.parser")

            review_text_tag = product_soup.select_one("span[data-hook='review-body']")
            review_text = review_text_tag.get_text(strip=True) if review_text_tag else ""

            # Sauvegarde des données
            titles.append(title)
            prices.append(price)
            ratings.append(rating)
            reviews.append(review_count)
            review_texts.append(review_text)

            print(f" Produit : {title[:50]}... Avis extrait : {len(review_text)} car.")

            time.sleep(1)

        except Exception as e:
            print(f" Erreur pour un produit : {e}")
            continue

# Création du DataFrame final
df = pd.DataFrame({
    "titre": titles,
    "prix": prices,
    "note": ratings,
    "nb_avis": reviews,
    "texte_avis": review_texts
})

# Supprimer les produits sans avis
df = df[df["texte_avis"] != ""]

# Sauvegarde
df.to_csv("amazon_avis_textes.csv", index=False)
print(f"\n Scraping terminé : {len(df)} produits enregistrés avec texte d’avis.")
