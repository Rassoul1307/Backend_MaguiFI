import numpy as np
import cv2
from typing import List, Tuple
from fastapi import UploadFile
from insightface.app import FaceAnalysis
from app.services.supabase_service import supabase  # ton client Supabase déjà initialisé

# --- Initialisation globale du modèle ---
face_app = None

def load_face_model():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


def get_embedding_and_crop_with_mask(image_bytes: bytes):
    app = load_face_model()

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    faces = app.get(img)
    if len(faces) == 0:
        return None

    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)

    # --- Sécuriser la découpe ---
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_crop = img[y1:y2, x1:x2].copy()

    # Vérifier si crop valide
    if face_crop.size == 0 or face_crop.shape[0] < 5 or face_crop.shape[1] < 5:
        # Crop trop petit, on ne dessine rien mais on garde embedding
        return face.embedding, img

    # --- Landmarks ---
    landmarks = face.landmark_2d_106.astype(int)

    # Décaler vers crop & clamp (IMPORTANT)
    def safe_points(points):
        pts = points - np.array([x1, y1])
        pts[:, 0] = np.clip(pts[:, 0], 0, face_crop.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, face_crop.shape[0] - 1)
        return pts

    try:
        # Contour visage
        jaw = safe_points(landmarks[0:17])
        cv2.polylines(face_crop, [jaw], True, (0,255,0), 2)

        # Yeux
        left_eye = safe_points(landmarks[36:42])
        right_eye = safe_points(landmarks[42:48])
        cv2.polylines(face_crop, [left_eye], True, (255,0,0), 2)
        cv2.polylines(face_crop, [right_eye], True, (255,0,0), 2)

        # Sourcils
        left_eyebrow = safe_points(landmarks[17:22])
        right_eyebrow = safe_points(landmarks[22:27])
        cv2.polylines(face_crop, [left_eyebrow], False, (0,255,255), 2)
        cv2.polylines(face_crop, [right_eyebrow], False, (0,255,255), 2)

        # Bouche
        mouth = safe_points(landmarks[48:68])
        cv2.polylines(face_crop, [mouth], True, (0,0,255), 2)

    except Exception:
        # On ignore le dessin si problème
        pass

    return face.embedding, face_crop


def compute_average_embedding_and_upload(photos: List[UploadFile], matricule: str) -> Tuple[list[float], List[str]] | None:

    """
    Calcule la moyenne des embeddings et upload les crops de visage dans Supabase.
    Retourne (avg_embedding, liste des URLs des crops).
    """
    embeddings = []
    crop_urls = []

    bucket = supabase.storage.from_("photos")

    for idx, photo in enumerate(photos):
        image_bytes = photo.file.read()
        result = get_embedding_and_crop_with_mask(image_bytes)

        if result is None:
            photo.file.seek(0)
            continue

        embedding, face_crop = result
        embeddings.append(embedding)

        # Convertir crop en JPEG bytes
        _, buffer = cv2.imencode(".jpg", face_crop)
        crop_bytes = buffer.tobytes()

        # Générer un chemin unique pour Supabase
        file_path = f"agents/{matricule}/face_{idx+1}.jpg"

        # Supprimer si existe déjà
        try:
            bucket.remove([file_path])
        except Exception:
            pass

        # Upload
        bucket.upload(file_path, crop_bytes, {"content-type": "image/jpeg"})
        crop_urls.append(bucket.get_public_url(file_path))

        # Remettre curseur pour lecture future si nécessaire
        photo.file.seek(0)

    if len(embeddings) == 0:
        return None

    avg_embedding = np.mean(np.stack(embeddings), axis=0).tolist()
    return avg_embedding, crop_urls



