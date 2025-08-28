from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import face_recognition
import numpy as np
import cv2
import io
from PIL import Image
import sqlite3
import json
from pydantic import BaseModel

app = FastAPI(
    title="Face Recognition API",
    description="API for registering and verifying users using face recognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL,
        face_encoding BLOB NOT NULL
    )
    """)
    conn.commit()
    conn.close()

init_db()

# Helper functions
def load_image_from_file(file):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

def get_face_encoding(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with only one face.")
    
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0]

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return distance <= tolerance

# Pydantic models
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: str

class VerificationResponse(BaseModel):
    verified: bool
    user: Optional[UserResponse]
    message: str

# API Endpoints
@app.post("/register", response_model=UserResponse, tags=["User Management"])
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Register a new user with face encoding
    """
    try:
        # Load and process image
        image = load_image_from_file(image)
        face_encoding = get_face_encoding(image)
        
        # Convert numpy array to bytes for storage
        encoding_bytes = face_encoding.tobytes()
        
        # Save to database
        conn = sqlite3.connect("face_recognition.db")
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (name, email, phone, face_encoding) VALUES (?, ?, ?, ?)",
                (name, email, phone, encoding_bytes)
            )
            user_id = cursor.lastrowid
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=400, detail="Email already registered")
        finally:
            conn.close()
        
        return UserResponse(id=user_id, name=name, email=email, phone=phone)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/verify", response_model=VerificationResponse, tags=["Face Verification"])
async def verify_user(image: UploadFile = File(...)):
    """
    Verify user identity using face recognition
    """
    try:
        # Load and process image
        image = load_image_from_file(image)
        unknown_encoding = get_face_encoding(image)
        
        # Get all users from database
        conn = sqlite3.connect("face_recognition.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, phone, face_encoding FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if not users:
            return VerificationResponse(
                verified=False,
                user=None,
                message="No registered users found"
            )
        
        # Compare with each user
        for user in users:
            user_id, name, email, phone, encoding_bytes = user
            known_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            
            if compare_faces(known_encoding, unknown_encoding):
                return VerificationResponse(
                    verified=True,
                    user=UserResponse(id=user_id, name=name, email=email, phone=phone),
                    message="Face verified successfully"
                )
        
        return VerificationResponse(
            verified=False,
            user=None,
            message="No matching face found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
