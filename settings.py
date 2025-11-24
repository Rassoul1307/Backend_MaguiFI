# settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Les variables sont lues automatiquement depuis l'environnement (Render)

    # Paramètres Supabase
    SUPABASE_URL: str = "https://miabadlxsxlgzgaoqdpv.supabase.co"
    SUPABASE_SECRET_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1pYWJhZGx4c3hsZ3pnYW9xZHB2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwMzY3NzIsImV4cCI6MjA3ODYxMjc3Mn0.-2dsje_RrVyQs6ri-J6jaC95kJEb737SWlofG6vreF8"
    SUPABASE_BUCKET: str = "agents-photos"

    # Paramètres FastAPI
    APP_NAME: str = "Mon API FastAPI"
    DEBUG: bool = False

# Instanciez la configuration une seule fois
settings = Settings()