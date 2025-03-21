import os
import pandas as pd
from scipy.stats import kendalltau, spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ Fichier introuvable : {DATA_PATH}")

print(f"✅ Chargement des données depuis : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"🔍 Nombre d'entrées dans le CSV : {df.shape[0]}")

df_aliments = df["aliments"].str.get_dummies(sep=", ")
df = pd.concat([df_aliments, df.drop(columns=["aliments"])], axis=1)

print("🔍 Analyse des corrélations avec Kendall :\n")
for col in ["bristol", "douleur", "odeur", "residus", "mucus", "flatulence", "ballonnement", "coliques", "sensation"]:
    corr, _ = kendalltau(df["bristol"], df[col])
    print(f"Corrélation entre Bristol Scale et {col}: {corr:.3f}")
