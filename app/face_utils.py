import face_recognition
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional

def load_image_from_file(file):
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)

def get_face_locations(image, model='cnn'):
    # Using CNN model for better accuracy with different orientations
    return face_recognition.face_locations(image, model=model)

def get_face_encodings(image, face_locations=None):
    if face_locations is None:
        face_locations = get_face_locations(image)
    return face_recognition.face_encodings(image, face_locations)

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return distance <= tolerance

def draw_boxes_and_names(image, face_locations, names):
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
    # Convert back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_to_base64(image):
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str