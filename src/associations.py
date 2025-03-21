import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ Fichier introuvable : {DATA_PATH}")

print(f"✅ Chargement des données depuis : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

transactions = df["aliments"].str.split(", ")

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_apriori = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_apriori, min_support=0.005, use_colnames=True)  # 🔽 Min support encore plus bas (0.5%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)  # 🔽 Confiance plus faible (20%)

rules = rules.sort_values(by="confidence", ascending=False)

if not rules.empty:
    print("🔍 Top 10 des associations alimentaires à risque :")
    print(rules[["antecedents", "consequents", "support", "confidence"]].head(10))
else:
    print("⚠️ Toujours aucune association significative trouvée. Peut-être que les aliments ne sont pas assez récurrents.")
