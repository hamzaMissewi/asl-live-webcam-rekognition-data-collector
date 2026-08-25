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

app = FastAPI(title="Gestures Detection Backend", description="Backend FastAPI pour le traducteur ASL en temps réel.", version="1.0.0")

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


_prev_face_expr = None
_prev_face_conf = 0.0


def classify_face_expression(landmarks):
    global _prev_face_expr, _prev_face_conf

    pts = np.array(landmarks).reshape(478, 3)

    MOUTH_LEFT, MOUTH_RIGHT = 61, 291
    MOUTH_TOP, MOUTH_BOTTOM = 13, 14
    LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
    LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
    RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 263, 362
    LEFT_EYEBROW_INNER, RIGHT_EYEBROW_INNER = 107, 336
    CHIN, FOREHEAD = 152, 10

    face_height = abs(pts[CHIN][1] - pts[FOREHEAD][1])
    if face_height < 0.001:
        face_height = 0.001

    mouth_width = abs(pts[MOUTH_LEFT][0] - pts[MOUTH_RIGHT][0])
    mouth_height = abs(pts[MOUTH_TOP][1] - pts[MOUTH_BOTTOM][1])
    mouth_center_y = (pts[MOUTH_TOP][1] + pts[MOUTH_BOTTOM][1]) / 2
    avg_corner_y = (pts[MOUTH_LEFT][1] + pts[MOUTH_RIGHT][1]) / 2

    mouth_curvature = avg_corner_y - mouth_center_y
    mar = mouth_height / mouth_width if mouth_width > 0.001 else 0

    left_eye_h = abs(pts[LEFT_EYE_TOP][1] - pts[LEFT_EYE_BOTTOM][1])
    left_eye_w = abs(pts[LEFT_EYE_LEFT][0] - pts[LEFT_EYE_RIGHT][0])
    right_eye_h = abs(pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYE_BOTTOM][1])
    right_eye_w = abs(pts[RIGHT_EYE_LEFT][0] - pts[RIGHT_EYE_RIGHT][0])
    avg_eye_h = (left_eye_h + right_eye_h) / 2
    avg_eye_w = (left_eye_w + right_eye_w) / 2
    ear = avg_eye_h / avg_eye_w if avg_eye_w > 0.001 else 0

    left_brow_dist = pts[LEFT_EYE_TOP][1] - pts[LEFT_EYEBROW_INNER][1]
    right_brow_dist = pts[RIGHT_EYE_TOP][1] - pts[RIGHT_EYEBROW_INNER][1]
    avg_brow_dist = (left_brow_dist + right_brow_dist) / 2

    norm_mouth_width = mouth_width / face_height
    norm_mouth_height = mouth_height / face_height
    norm_curvature = mouth_curvature / face_height
    norm_brow_dist = avg_brow_dist / face_height

    is_smile = norm_curvature < -0.005 and norm_mouth_width > 0.06
    is_wide_smile = norm_curvature < -0.01 and norm_mouth_width > 0.12 and mar > 0.15
    is_open_mouth = mar > 0.25
    is_squinting = ear < 0.18
    is_wide_eyes = ear > 0.38
    is_brows_raised = norm_brow_dist > 0.07
    is_frown = norm_curvature > 0.006 and norm_mouth_width > 0.06

    if is_wide_smile and is_open_mouth:
        expr, emoji, conf = "happy", "😊", 0.92
    elif is_squinting and is_open_mouth and is_frown:
        expr, emoji, conf = "cry", "😢", 0.85
    elif is_smile:
        expr, emoji, conf = "smile", "😄", 0.88
    elif is_frown and is_brows_raised:
        expr, emoji, conf = "sad", "😞", 0.8
    elif is_wide_eyes and is_brows_raised and is_open_mouth:
        expr, emoji, conf = "surprise", "😲", 0.9
    elif is_squinting and not is_open_mouth:
        expr, emoji, conf = "sad", "😞", 0.75
    elif is_frown:
        expr, emoji, conf = "sad", "😞", 0.7
    else:
        expr, emoji, conf = "neutral", "😐", 0.8

    if expr == _prev_face_expr:
        conf = 0.6 * _prev_face_conf + 0.4 * conf
    else:
        if conf < 0.85 and _prev_face_conf > 0.6:
            expr = _prev_face_expr
            conf = _prev_face_conf * 0.8
    _prev_face_expr = expr
    _prev_face_conf = conf

    return {"expression": expr, "emoji": emoji, "confidence": round(conf, 2)}


@app.websocket("/ws-face")
async def ws_face(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            landmarks = payload.get("landmarks")

            if not landmarks or len(landmarks) != 1434:
                await websocket.send_json({"error": "landmarks visage invalides (attendu: 1434 valeurs)."})
                continue

            result = classify_face_expression(landmarks)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


def classify_hand_gesture(landmarks, history=None):
    pts = np.array(landmarks).reshape(21, 3)

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    THUMB_TIP, THUMB_IP = 4, 3

    fingers_up = []
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers_up.append(bool(pts[tip][1] < pts[pip][1]))

    thumb_tip_x = pts[THUMB_TIP][0]
    thumb_ip_x = pts[THUMB_IP][0]
    palm_center_x = pts[9][0]
    thumb_extended = abs(thumb_tip_x - palm_center_x) > abs(thumb_ip_x - palm_center_x) * 1.1

    total = sum(fingers_up) + (1 if thumb_extended else 0)

    if total == 0:
        gesture, emoji = "poing", "✊"
    elif total == 5 and all(fingers_up):
        gesture, emoji = "main ouverte", "🖐️"
    elif fingers_up[0] and not any(fingers_up[1:]) and not thumb_extended:
        gesture, emoji = "pointe", "☝️"
    elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        gesture, emoji = "paix", "✌️"
    elif thumb_extended and not any(fingers_up):
        gesture, emoji = "pouce en l'air", "👍"
    elif fingers_up[0] and fingers_up[1] and fingers_up[2] and not fingers_up[3]:
        gesture, emoji = "three", "🤟"
    elif not fingers_up[0] and fingers_up[1] and fingers_up[2] and not fingers_up[3]:
        gesture, emoji = "rock", "🤘"
    else:
        gesture, emoji = f"{total} doigts", "🖐️"

    movement = "statique"
    speed = 0.0
    if history and len(history) >= 3:
        recent = np.array(history[-5:]) if len(history) >= 5 else np.array(history)
        if recent.shape[0] >= 2:
            dx = np.mean(np.diff(recent[:, 0]))
            dy = np.mean(np.diff(recent[:, 1]))
            speed = float(np.sqrt(dx ** 2 + dy ** 2))
            if speed > 0.015:
                if abs(dx) > abs(dy):
                    movement = "gauche" if dx > 0 else "droite"
                else:
                    movement = "haut" if dy > 0 else "bas"

    return {
        "gesture": gesture,
        "emoji": emoji,
        "fingers": total,
        "movement": movement,
        "speed": round(speed, 4),
        "confidence": 0.85,
    }


@app.websocket("/ws-gesture")
async def ws_gesture(websocket: WebSocket):
    await websocket.accept()
    try:
        history = []
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            landmarks = payload.get("landmarks")

            if not landmarks or len(landmarks) != 63:
                await websocket.send_json({"error": "landmarks invalides (attendu: 63 valeurs)."})
                continue

            wrist = landmarks[0:3]
            history.append(wrist)
            if len(history) > 10:
                history = history[-10:]

            result = classify_hand_gesture(landmarks, history)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
