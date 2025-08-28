# Face Recognition API

A FastAPI-based face recognition system that allows registering users with their face images and verifying their identity through face recognition.

## Features

- User registration with face encoding
- Face verification against registered users
- Automatic face detection and encoding
- SQLite database for user storage
- Docker support for easy deployment
- Swagger UI for API documentation

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Face Recognition**: face_recognition, OpenCV, Pillow
- **Database**: SQLite (with SQLAlchemy)
- **Deployment**: Docker, Docker Compose

## Installation

### Prerequisites

- Python 3.9+
- pip

### System Dependencies

The `face_recognition` library requires dlib. Install the following system dependencies:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libjpeg-dev libpng-dev
