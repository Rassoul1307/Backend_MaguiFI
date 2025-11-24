# download_models.py
import insightface
from insightface.app import FaceAnalysis
import os

# IMPORTANT : InsightFace télécharge le modèle dans le répertoire HOME de l'utilisateur.
# Sur Render, le répertoire de travail est /opt/render/project/src/.
# Si nous ne changeons rien, le modèle ira dans /opt/render/.insightface
# Pour que l'étape de Build le trouve, nous initialisons simplement FaceAnalysis.

print("-> Démarrage du pré-téléchargement du modèle InsightFace 'buffalo_l'...")

try:
    # L'appel à l'instanciation force le téléchargement du modèle s'il n'existe pas.
    # Nous n'avons besoin de rien faire avec l'objet, juste que la ligne s'exécute.
    app = FaceAnalysis(name='buffalo_l')
    
    print("-> Pré-téléchargement du modèle 'buffalo_l' terminé avec succès.")
    
except Exception as e:
    # Ceci est critique. Le build doit réussir. Si le téléchargement échoue (rare), 
    # nous loguons mais permettons au build de se terminer.
    print(f"ATTENTION: Échec du pré-téléchargement du modèle : {e}")
    
# Le modèle est maintenant stocké de manière permanente dans l'image de conteneur Render.