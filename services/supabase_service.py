# app/services/supabase_service.py
import os
from typing import List, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import UploadFile
import time
import io

# Charger .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Nom du bucket stockage où tu veux mettre les photos (crée-le depuis Supabase UI)
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "agents-photos")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")

# Création du client Supabase (sync)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _make_path(matricule: str, filename: str) -> str:
    """
    Génère un chemin unique dans le bucket pour éviter collisions.
    Exemple: "matricule/1699999999_filename.jpg"
    """
    ts = int(time.time() * 1000)
    safe_name = filename.replace(" ", "_")
    return f"{matricule}/{ts}_{safe_name}"



def upload_photos_to_supabase(matricule: str, photos):
    """
    Upload les photos dans Supabase Storage
    et retourne une liste d'URLs publiques.
    """
    folder_name = f"agents/{matricule}"
    bucket = supabase.storage.from_("photos")
    photo_urls = []

    for idx, photo in enumerate(photos):
        file_bytes = photo.file.read()
        file_path = f"{folder_name}/photo_{idx+1}.jpg"

        try:
            # --- Supprimer si le fichier existe déjà (optionnel) ---
            try:
                bucket.remove([file_path])
            except Exception:
                pass  # ignore si le fichier n'existe pas

            # --- Upload du fichier ---
            response = bucket.upload(
                file_path,
                file_bytes,
                {"content-type": "image/jpeg"}
            )

            # --- Vérifier s'il y a une erreur ---
            if isinstance(response, dict) and response.get("error"):
                raise Exception(response["error"])

            # --- Obtenir l'URL publique ---
            public_url = bucket.get_public_url(file_path)
            photo_urls.append(public_url)

        except Exception as e:
            raise Exception(f"Erreur upload {file_path}: {e}")

    return photo_urls


def get_public_url(path: str) -> str:
    """
    Retourne l'URL publique (convenience) pour un objet dans un bucket public.
    -> Assure-toi que le bucket est _public_ dans Supabase Storage, sinon utilise create_signed_url.
    """
    response = supabase.storage.from_(BUCKET_NAME).get_public_url(path)
    # selon la lib, response peut être un dict {'publicUrl': '...'} ou {'data': {'publicUrl': '...'}}
    # mais l'API docs montre un return simple — on va être prudent :
    if isinstance(response, dict):
        # cas: {'publicUrl': '...'} ou {'data': {'publicUrl': '...'}}
        if "publicUrl" in response:
            return response["publicUrl"]
        if "data" in response and isinstance(response["data"], dict) and "publicUrl" in response["data"]:
            return response["data"]["publicUrl"]
    # fallback : construire l'URL manuellement (moins recommandé)
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{path}"


def create_agent_record(
    nom: str,
    prenom: str,
    matricule: str,
    service: str,
    telephone: str | None,
    photo_paths: List[str],
    embedding: List[float],
    status: str = "en_attente",
) -> dict:
    """
    Insert un enregistrement dans la table `agents`.
    - photo_paths: liste de paths (strings) dans le bucket (stocker ces paths en DB, pas l'URL).
    - embedding: liste de floats (par ex. moyenne des embeddings)
    Retourne la réponse du supabase insert.
    """
    record = {
        "nom": nom,
        "prenom": prenom,
        "matricule": matricule,
        "service": service,
        "telephone": telephone,
        "photos": photo_paths,      # text[] dans ta table
        "embedding": embedding,     # vector(512) ou float8[] selon ta config
        "status": status,
    }
    res = supabase.table("agents").insert(record).execute()
    # res contient .data et .error selon la lib; retourne le tout
    return res