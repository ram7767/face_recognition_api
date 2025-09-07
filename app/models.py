from pydantic import BaseModel
from typing import Optional, List

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: str

class VerificationResponse(BaseModel):
    verified: bool
    user: Optional[UserResponse]
    message: str
    imageWithBoxes: Optional[str] = None  # base64 encoded image

class FaceBox(BaseModel):
    top: int
    right: int
    bottom: int
    left: int
    name: str

class MultipleFacesResponse(BaseModel):
    faces: List[FaceBox]
    imageWithBoxes: str  # base64 encoded image
    message: str