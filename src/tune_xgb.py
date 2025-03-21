import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [200, 500, 800],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
    "gamma": [0, 0.1, 0.3],
}

model = xgb.XGBClassifier(eval_metric="mlogloss")

grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"✅ Meilleur XGBoost : {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
print(f"🎯 Précision du meilleur XGBoost : {accuracy_score(y_test, y_pred):.2f}")

model_path = os.path.join(BASE_DIR, "models", "xgboost_optimized.pkl")
import pickle
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"✅ Modèle XGBoost optimisé sauvegardé dans : {model_path}")
