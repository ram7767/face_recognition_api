import face_recognition
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import io
from PIL import Image
import sqlite3
import numpy as np
from app.models import UserResponse, VerificationResponse, MultipleFacesResponse, FaceBox
from app.database import get_db_connection, init_db
from app.face_utils import (
    load_image_from_file, 
    get_face_locations, 
    get_face_encodings, 
    compare_faces, 
    draw_boxes_and_names,
    image_to_base64
)
from app.config import settings

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
init_db()

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
        image_array = load_image_from_file(image)
        face_locations = get_face_locations(image_array)
        
        if len(face_locations) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the image. Please provide a clear image with a visible face."
            )
        if len(face_locations) > 1:
            # We'll return an image with boxes and names for multiple faces
            face_encodings = get_face_encodings(image_array, face_locations)
            names = ["Unknown"] * len(face_encodings)
            imageWithBoxes = draw_boxes_and_names(image_array, face_locations, names)
            img_base64 = image_to_base64(imageWithBoxes)
            
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Multiple faces detected. Please provide an image with only one face.",
                    "imageWithBoxes": img_base64
                }
            )
        
        face_encoding = get_face_encodings(image_array, face_locations)[0]
        
        # Convert numpy array to bytes for storage
        encoding_bytes = face_encoding.tobytes()
        
        # Save to database
        conn = get_db_connection()
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
            raise HTTPException(
                status_code=400, 
                detail="Email already registered. Please use a different email address."
            )
        finally:
            conn.close()
        
        return UserResponse(id=user_id, name=name, email=email, phone=phone)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}. Please ensure the image is in a supported format (JPEG, PNG)."
        )

@app.post("/verify", response_model=VerificationResponse, tags=["Face Verification"])
async def verify_user(image: UploadFile = File(...)):
    """
    Verify user identity using face recognition
    """
    try:
        # Load and process image
        image_array = load_image_from_file(image)
        face_locations = get_face_locations(image_array)
        
        if len(face_locations) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the image. Please provide a clear image with a visible face."
            )
        
        face_encodings = get_face_encodings(image_array, face_locations)
        
        # Get all users from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, phone, face_encoding FROM users")
        users = cursor.fetchall()
        conn.close()
        
        if not users:
            # No users in database, so we return the image with boxes and "Unknown" for each face
            names = ["Unknown"] * len(face_encodings)
            imageWithBoxes = draw_boxes_and_names(image_array, face_locations, names)
            img_base64 = image_to_base64(imageWithBoxes)
            
            return VerificationResponse(
                verified=False,
                user=None,
                message="No registered users found. Please register a user first.",
                imageWithBoxes=img_base64
            )
        
        # Prepare known encodings and user data
        known_encodings = []
        user_data = []
        for user in users:
            user_id, name, email, phone, encoding_bytes = user
            known_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            known_encodings.append(known_encoding)
            user_data.append({"id": user_id, "name": name, "email": email, "phone": phone})
        
        # Compare each face in the image with known encodings
        names = []
        matched_user = None
        for face_encoding in face_encodings:
            # Check against all known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            
            if min_distance <= 0.6:  # tolerance
                matched_user = user_data[min_distance_index]
                names.append(matched_user["name"])
            else:
                names.append("Unknown")
        
        # Draw boxes and names on the image
        imageWithBoxes = draw_boxes_and_names(image_array, face_locations, names)
        img_base64 = image_to_base64(imageWithBoxes)
        
        if matched_user:
            return VerificationResponse(
                verified=True,
                user=UserResponse(
                    id=matched_user["id"],
                    name=matched_user["name"],
                    email=matched_user["email"],
                    phone=matched_user["phone"]
                ),
                message="Face verified successfully",
                imageWithBoxes=img_base64
            )
        else:
            return VerificationResponse(
                verified=False,
                user=None,
                message="No matching face found. Please register first or try with a better image.",
                imageWithBoxes=img_base64
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}. Please ensure the image is in a supported format (JPEG, PNG)."
        )

@app.post("/detect-faces", response_model=MultipleFacesResponse, tags=["Face Detection"])
async def detect_faces(image: UploadFile = File(...)):
    """
    Detect faces in the image and return bounding boxes with names if recognized
    """
    try:
        # Load and process image
        image_array = load_image_from_file(image)
        face_locations = get_face_locations(image_array)
        
        if len(face_locations) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the image. Please provide a clear image with a visible face."
            )
        
        face_encodings = get_face_encodings(image_array, face_locations)
        
        # Get all users from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, phone, face_encoding FROM users")
        users = cursor.fetchall()
        conn.close()
        
        names = []
        face_boxes = []
        
        if users:
            # Prepare known encodings and user data
            known_encodings = []
            user_data = []
            for user in users:
                user_id, name, email, phone, encoding_bytes = user
                known_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
                known_encodings.append(known_encoding)
                user_data.append({"id": user_id, "name": name, "email": email, "phone": phone})
            
            # Compare each face in the image with known encodings
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                
                if min_distance <= 0.6:  # tolerance
                    matched_user = user_data[min_distance_index]
                    names.append(matched_user["name"])
                else:
                    names.append("Unknown")
        else:
            names = ["Unknown"] * len(face_encodings)
        
        # Create face boxes
        for (top, right, bottom, left), name in zip(face_locations, names):
            face_boxes.append(FaceBox(top=top, right=right, bottom=bottom, left=left, name=name))
        
        # Draw boxes and names on the image
        imageWithBoxes = draw_boxes_and_names(image_array, face_locations, names)
        img_base64 = image_to_base64(imageWithBoxes)
        
        return MultipleFacesResponse(
            faces=face_boxes,
            imageWithBoxes=img_base64,
            message=f"Detected {len(face_locations)} faces"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}. Please ensure the image is in a supported format (JPEG, PNG)."
        )

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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)