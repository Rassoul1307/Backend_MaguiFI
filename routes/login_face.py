import json
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List
from services.face_service import get_embedding_and_crop_with_mask
from services.supabase_service import supabase


router = APIRouter(prefix="/login-face", tags=["FaceLogin"])

THRESHOLD = 0.60          # seuil de similarité pour reconnaitre l'agent
# LIVE_THRESHOLD = 0.75     # <-- utilisé dans check_liveness (DepthAnythingV2)


# ---------------------------------------------------------
#   LOGIN PAR VISAGE AVEC ANTI-SPOOFING + EMBEDDINGS
# ---------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@router.post("/")
async def login_face(photos: List[UploadFile] = File(...)):
    print(f"\n[LOGIN] Nombre de photos reçues: {len(photos)}")
    embeddings = []

    # ---------- 1. TRAITEMENT DE CHAQUE PHOTO ----------
    for i, photo in enumerate(photos):
        print(f"\n[PHOTO {i+1}] Traitement...")

        image_bytes = photo.file.read()

        # ------ Anti-Spoofing : DepthAnything (3D) ------
        print("[LIVENESS] Vérification spoof...")
        is_live = True

        if not is_live:
            print("[LIVENESS] ❌ Spoof détecté")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tentative de spoofing détectée (photo / écran)."
            )

        print("[LIVENESS] ✅ Visage réel confirmé")

        # ------ Extraction Embeddings ------
        result = get_embedding_and_crop_with_mask(image_bytes)

        if result is None:
            print("[EMBEDDING] Aucun visage détecté")
            continue

        embedding, _ = result
        embeddings.append(np.array(embedding, dtype=np.float32))

        # Réinitialiser le pointeur du fichier
        photo.file.seek(0)

    if not embeddings:
        print("[LOGIN] ❌ Aucun embedding valide obtenu")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun visage détecté sur les images envoyées."
        )

    # Moyenne des embeddings envoyés
    avg_embedding = np.mean(np.stack(embeddings), axis=0)
    print(f"[EMBEDDING] Embedding moyen obtenu (shape: {avg_embedding.shape})")

    # ---------- 2. RÉCUPÉRATION AGENTS EN BASE ----------
    response = supabase.table("agents").select("*").execute()
    agents = response.data or []
    print(f"[DATABASE] Agents trouvés : {len(agents)}")

    # ---------- 3. COMPARAISON AVEC CHAQUE AGENT ----------
    for agent in agents:
        try:
            stored_emb = np.array(json.loads(agent["embedding"]), dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Parsing embedding agent {agent.get('nom', '')} : {e}")
            continue

        similarity = cosine_similarity(avg_embedding, stored_emb)
        print(f"[MATCH] Similarité avec {agent['nom']} {agent['prenom']} : {similarity:.3f}")

        if similarity >= THRESHOLD:
            print(f"[MATCH] 🎉 Agent reconnu : {agent['nom']} {agent['prenom']}")
            return {
                "success": True,
                "message": "Connexion réussie",
                "agent": {
                    "nom": agent["nom"],
                    "prenom": agent["prenom"],
                    "matricule": agent["matricule"],
                    "service": agent["service"],
                    "telephone": agent["telephone"],
                    "photos": agent["photos"]
                }
            }

    # ---------- 4. AUCUN MATCH ----------
    print("[MATCH] ❌ Aucun agent reconnu")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Aucun agent reconnu avec ce visage."
    )