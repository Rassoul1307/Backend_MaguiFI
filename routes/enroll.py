# app/routes/enroll.py

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from typing import List
from services.supabase_service import upload_photos_to_supabase, create_agent_record
# from app.services.face_service import compute_average_embedding
from services.face_service import compute_average_embedding_and_upload

router = APIRouter(prefix="/enroll", tags=["Enrollment"])

@router.post("/")
async def enroll_agent(
    nom: str = Form(...),
    prenom: str = Form(...),
    matricule: str = Form(...),
    service: str = Form(...),
    telephone: str = Form(...),
    photos: List[UploadFile] = File(...),
):
    """
    Endpoint d'enrôlement d'un agent.
    """

    try:
        # 1️⃣ Calcul des embeddings
        avg_embedding, crop_urls = compute_average_embedding_and_upload(photos, matricule)
        if avg_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Aucun visage détecté sur les photos envoyées."
            )

        # 2️⃣ Upload des photos
        photo_urls = upload_photos_to_supabase(matricule, photos)

        # 3️⃣ Insertion dans Supabase
        response = create_agent_record(
            nom=nom,
            prenom=prenom,
            matricule=matricule,
            service=service,
            telephone=telephone,
            photo_paths=photo_urls,
            embedding=avg_embedding,
            status="en_attente"
        )

        return {
            "success": True,
            "message": "Agent enrôlé avec succès.",
            "data": response.data if hasattr(response, "data") else response
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )