from fastapi import FastAPI
from routes import enroll
from routes import login_face
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agent Enrollment API")

# Autoriser ton front (ici tout pour tester)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour tester on accepte tout, ensuite restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure la route
app.include_router(enroll.router)
app.include_router(login_face.router)