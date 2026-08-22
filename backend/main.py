"""
Backend FastAPI pour le traducteur ASL en temps réel.

Reçoit des landmarks de main (21 points x,y,z envoyés par le frontend via
WebSocket), les fait passer dans un modèle entraîné (voir train_model.py),
et renvoie la prédiction + un score de confiance.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

MODEL_PATH = Path(__file__).parent / "model.joblib"
LABELS_PATH = Path(__file__).parent / "labels.json"

app = FastAPI(title="ASL Live Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
labels = None


def load_model():
    """Charge le modèle et la liste des labels s'ils existent."""
    global model, labels
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        labels = json.loads(LABELS_PATH.read_text())
        print(f"Modèle chargé — {len(labels)} classes : {labels}")
    else:
        model = None
        labels = None
        print("Aucun modèle trouvé. Lance train_model.py après avoir exporté un dataset.")


load_model()


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_classes": len(labels) if labels else 0,
    }


@app.post("/reload-model")
def reload_model():
    """Permet de recharger le modèle sans redémarrer le serveur (après un ré-entraînement)."""
    load_model()
    return {"reloaded": True, "model_loaded": model is not None}


@app.websocket("/ws")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            landmarks = payload.get("landmarks")

            if model is None:
                await websocket.send_json(
                    {"error": "Aucun modèle entraîné — lance train_model.py d'abord."}
                )
                continue

            if not landmarks or len(landmarks) != 63:
                await websocket.send_json({"error": "landmarks invalides (attendu: 63 valeurs)."})
                continue

            x = np.array(landmarks).reshape(1, -1)
            proba = model.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            await websocket.send_json(
                {"label": labels[idx], "confidence": float(proba[idx])}
            )
    except WebSocketDisconnect:
        pass
