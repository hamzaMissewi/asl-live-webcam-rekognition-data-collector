# ASL Live — Traducteur de langue des signes en temps réel

Projet complet (V1) : collecte de données, entraînement, et traduction ASL en
temps réel via webcam. Reconnaît l'alphabet ASL (A-Z) + quelques mots
courants (hello, thanks, yes, no, please), extensible facilement.

## Architecture

```
frontend/index.html   → Webcam + MediaPipe Hands (extraction de landmarks,
                         100% dans le navigateur) + dashboard
backend/main.py        → FastAPI + WebSocket, charge le modèle et
                         renvoie une prédiction en temps réel
backend/train_model.py → Entraîne un MLP sur les landmarks collectés
```

Rien ne transite d'autre que des coordonnées de points (63 floats par
frame) entre le navigateur et le serveur — pas de flux vidéo envoyé, donc
latence très faible.

## Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate  # (venv\Scripts\activate sur Windows)
pip install -r requirements.txt
```

## Workflow (à suivre dans l'ordre)

### 1. Collecter des données

1. Ouvre `frontend/index.html` directement dans ton navigateur (double-clic,
   ou `python -m http.server` depuis le dossier `frontend/` puis
   `http://localhost:8000`).
2. Autorise l'accès à la webcam.
3. Onglet **« 1 · Collecter des données »** : choisis une étiquette (ex. « A »),
   positionne ta main, clique sur **« Enregistrer 60 frames »**. Répète pour
   chaque lettre/mot — idéalement plusieurs prises, avec de légères variations
   d'angle et de distance (vise au moins 60-100 échantillons par classe).
4. Clique sur **« Exporter le dataset (CSV) »** → télécharge
   `landmarks_dataset.csv`. Place ce fichier dans `backend/`.

### 2. Entraîner le modèle

```bash
cd backend
python train_model.py --data landmarks_dataset.csv
```

Ça produit `model.joblib` et `labels.json` dans `backend/`, et affiche un
rapport de performance par classe.

### 3. Lancer le backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Vérifie que ça tourne : `http://localhost:8000` doit répondre
`{"status": "ok", "model_loaded": true, ...}`.

### 4. Traduire en direct

Retourne dans `frontend/index.html`, onglet **« 2 · Traduire en direct »**,
clique sur **« Se connecter au backend »**, puis signe devant la caméra. Le
dashboard affiche la lettre/mot prédit, un score de confiance (jauge à
points), et l'historique des dernières prédictions.

## Étendre le projet (V2)

- **Plus de vocabulaire** : ajoute des étiquettes dans le tableau `LABELS` du
  frontend, recollecte, ré-entraîne.
- **Signes dynamiques (mots à mouvement)** : passer d'un modèle par-frame à
  un modèle séquentiel (LSTM/GRU) sur une fenêtre glissante de landmarks —
  nécessite d'enregistrer des séquences plutôt que des frames isolées.
- **Deux mains / pose du corps** : passer de `Hands` à `Holistic` dans
  MediaPipe pour capturer aussi le buste, utile pour les signes ASL qui
  impliquent le visage ou les deux mains.
- **Déploiement** : le backend est un FastAPI standard, déployable tel quel
  (Render, Fly.io, etc.) ; il faudra alors changer l'URL du WebSocket dans le
  frontend (`ws://localhost:8000/ws` → l'URL de prod, en `wss://`).

## Notes

- Le modèle MLP tourne sur des landmarks (63 nombres), pas sur des pixels :
  entraînement rapide (quelques secondes à minutes) et pas besoin de GPU.
- La qualité dépend directement de la quantité et de la diversité des
  données collectées à l'étape 1 — c'est le vrai levier si les prédictions
  sont peu fiables.
