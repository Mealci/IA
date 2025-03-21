import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ Fichier introuvable : {DATA_PATH}")

print(f"✅ Chargement des données depuis : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df_aliments = df["aliments"].str.get_dummies(sep=", ")
df = pd.concat([df_aliments, df.drop(columns=["aliments"])], axis=1)

X = df.drop(columns=["sensation"])
y = df["sensation"] - 1

print("🔍 Distribution des labels (sensation) :")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.01,
    eval_metric="mlogloss"
)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"📊 Validation croisée - Moyenne de précision : {cv_scores.mean():.2f} (± {cv_scores.std():.2f})")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"🎯 Précision finale du modèle : {accuracy_score(y_test, y_pred):.2f}")

plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.show()

model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Modèle entraîné et sauvegardé dans : {model_path}")