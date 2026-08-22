"""
Entraîne un classifieur léger (MLP) sur les landmarks de main exportés
depuis le frontend (bouton « Exporter le dataset (CSV) »).

Usage :
    python train_model.py --data landmarks_dataset.csv

Produit deux fichiers dans backend/ :
    - model.joblib   (le modèle entraîné)
    - labels.json    (la liste ordonnée des classes, dans l'ordre des indices du modèle)
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main(data_path: str):
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'label'.")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    labels = sorted(df["label"].unique().tolist())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])

    counts = df["label"].value_counts()
    print("Échantillons par classe :")
    print(counts)
    if counts.min() < 10:
        print(
            "\n⚠ Certaines classes ont très peu d'échantillons (<10). "
            "Le modèle sera peu fiable pour elles — recollecte des données si besoin.\n"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx if counts.min() >= 2 else None
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("\nRapport sur le jeu de test :")
    print(classification_report(y_test, clf.predict(X_test), target_names=labels, zero_division=0))

    out_dir = Path(__file__).parent
    joblib.dump(clf, out_dir / "model.joblib")
    (out_dir / "labels.json").write_text(json.dumps(labels))

    print(f"\nModèle sauvegardé : {out_dir / 'model.joblib'}")
    print(f"Labels sauvegardés : {out_dir / 'labels.json'}")
    print("\nRedémarre le backend (ou appelle POST /reload-model) pour charger le nouveau modèle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="landmarks_dataset.csv", help="Chemin vers le CSV exporté depuis le frontend")
    args = parser.parse_args()
    main(args.data)
