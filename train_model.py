"""Train the PCOS-risk screening model.

By default uses a SYNTHETIC dataset (so the repo runs anywhere).
To use a real dataset (e.g. Kaggle "Polycystic ovary syndrome (PCOS)" by
Prasoon Kottarathil), put a CSV at data/pcos.csv with the columns in FEATURES
plus a binary column `label`, then run:  python -m src.train_model --real
"""
import argparse, json, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

FEATURES = ["age", "bmi", "cycle_length_days", "cycles_per_year", "weight_gain",
            "hair_growth", "acne", "skin_darkening", "hair_loss", "fast_food_freq",
            "exercise_days"]
LABELS = {
    "age": "Age", "bmi": "BMI", "cycle_length_days": "Average cycle length (days)",
    "cycles_per_year": "Periods per year", "weight_gain": "Recent unexplained weight gain",
    "hair_growth": "Excess facial/body hair", "acne": "Persistent acne",
    "skin_darkening": "Skin darkening (neck/underarms)", "hair_loss": "Scalp hair thinning",
    "fast_food_freq": "Fast food days / week", "exercise_days": "Exercise days / week",
}

def synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.25, n)
    d = pd.DataFrame({"label": y})
    d["age"] = rng.integers(15, 45, n)
    d["bmi"] = np.clip(rng.normal(22, 3, n) + 4.5 * y, 16, 42).round(1)
    d["cycle_length_days"] = np.clip(rng.normal(29, 2.5, n) + 9 * y * rng.random(n), 20, 90).round()
    d["cycles_per_year"] = np.clip(np.round(365 / d["cycle_length_days"] - 3 * y * rng.random(n)), 2, 14)
    for col, p0, p1 in [("weight_gain", .15, .6), ("hair_growth", .08, .55), ("acne", .25, .6),
                        ("skin_darkening", .06, .45), ("hair_loss", .12, .5)]:
        d[col] = rng.binomial(1, np.where(y == 1, p1, p0))
    d["fast_food_freq"] = np.clip(rng.poisson(1.5 + 1.5 * y), 0, 7)
    d["exercise_days"] = np.clip(rng.poisson(3 - 1.2 * y), 0, 7)
    return d[FEATURES + ["label"]]

def main(real=False):
    df = pd.read_csv("data/pcos.csv") if real else synthetic()
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    print(classification_report(yte, model.predict(Xte)))
    print(f"ROC-AUC: {auc:.3f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/pcos_rf.joblib")
    meta = {"roc_auc": round(float(auc), 3), "n_train": int(len(Xtr)),
            "data": "real" if real else "synthetic",
            "importances": dict(zip(FEATURES, map(float, model.feature_importances_)))}
    json.dump(meta, open("models/meta.json", "w"), indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true")
    main(ap.parse_args().real)
