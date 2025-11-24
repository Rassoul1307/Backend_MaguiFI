import numpy as np
import cv2
from typing import List, Tuple
from fastapi import UploadFile
from insightface.app import FaceAnalysis
from services.supabase_service import supabase  # ton client Supabase déjà initialisé

# --- Initialisation globale du modèle ---
face_app = None

def load_face_model():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(name="buffalo_s")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app



def get_embedding_and_crop_with_mask(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Détecte le visage, applique un masque AR basé sur les landmarks, 
    retourne (embedding, crop avec masque).
    None si aucun visage détecté.
    """
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
    face_crop = img[y1:y2, x1:x2].copy()  # copie pour masque

    # --- Dessiner masque AR ---
    landmarks = face.landmark_2d_106  # (106,2)

    # Exemple simple : relier contour visage (points 0-16) + yeux + bouche
    # Contour visage
    jaw = landmarks[0:17].astype(int)
    cv2.polylines(face_crop, [jaw - [x1, y1]], isClosed=True, color=(0, 255, 0), thickness=2)

    # Yeux (approx 36-41 et 42-47)
    left_eye = landmarks[36:42].astype(int)
    right_eye = landmarks[42:48].astype(int)
    cv2.polylines(face_crop, [left_eye - [x1, y1]], True, (255,0,0), 2)
    cv2.polylines(face_crop, [right_eye - [x1, y1]], True, (255,0,0), 2)

    # Sourcils (17-21 et 22-26)
    left_eyebrow = landmarks[17:22].astype(int)
    right_eyebrow = landmarks[22:27].astype(int)
    cv2.polylines(face_crop, [left_eyebrow - [x1, y1]], False, (0,255,255), 2)
    cv2.polylines(face_crop, [right_eyebrow - [x1, y1]], False, (0,255,255), 2)

    # Bouche (48-67)
    mouth = landmarks[48:68].astype(int)
    cv2.polylines(face_crop, [mouth - [x1, y1]], True, (0,0,255), 2)

    embedding = face.embedding
    return embedding, face_crop



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



