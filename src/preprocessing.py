import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.json")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

print(f"✅ Chargement des données depuis : {DATA_PATH}")

def extract_aliments(entry):
    """ Gère les différents formats possibles de `aliments_ingeres` """
    if isinstance(entry["aliments_ingeres"], list):
        return ", ".join(str(item) for item in entry["aliments_ingeres"])
    elif isinstance(entry["aliments_ingeres"], dict):
        return ", ".join(entry["aliments_ingeres"].values())
    else:
        return str(entry["aliments_ingeres"])

def load_data(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 Nombre d'entrées chargées avant correction : {len(data)}")
    print("DEBUG - Type de `data` :", type(data))
    print("DEBUG - Exemple d'une entrée complète :", json.dumps(data[0], indent=4))

    flattened_data = []
    for entry in data:
        if isinstance(entry, list):
            flattened_data.extend(entry)
        else:
            flattened_data.append(entry)

    print(f"✅ Nombre d'entrées après correction : {len(flattened_data)}")

    rows = []
    for entry in flattened_data:
        if "aliments_ingeres" not in entry or "excrements" not in entry:
            print(f"⚠️ Entrée mal formée ignorée: {entry}")
            continue

        aliments = extract_aliments(entry)

        for excrement in entry["excrements"]:
            bristol = excrement.get("bristol_scale", None)
            douleur = excrement.get("douleur", None)
            odeur = excrement.get("odeur", None)
            residus = excrement.get("residus_alimentaires", None)
            mucus = excrement.get("mucus", None)
            flatulence = excrement.get("flatulence", None)
            ballonnement = excrement.get("ballonnement", None)
            coliques = excrement.get("coliques", None)
            sensation = entry.get("sensation", None)

            rows.append([
                aliments, bristol, douleur, odeur, residus, mucus, 
                flatulence, ballonnement, coliques, sensation
            ])

    df = pd.DataFrame(rows, columns=[
        "aliments", "bristol", "douleur", "odeur", "residus", "mucus", 
        "flatulence", "ballonnement", "coliques", "sensation"
    ])
    
    return df

df = load_data(DATA_PATH)
df.to_csv(os.path.join(BASE_DIR, "data", "preprocessed_data.csv"), index=False)

print("✅ Données prétraitées et sauvegardées avec succès !")
