import os

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./face_recognition.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

settings = Settings()
