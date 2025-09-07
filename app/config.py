import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./face_recognition.db")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

settings = Settings()