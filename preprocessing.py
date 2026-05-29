import cv2
import numpy as np

IMG_SIZE = (100, 100)

def load_and_preprocess(image_path=None, image_array=None):
    """Load gambar dari path atau array, return vektor."""
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = image_array
    
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca")
    
    # Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize
    resized = cv2.resize(gray, IMG_SIZE)
    
    # Normalisasi
    normalized = resized / 255.0
    
    # Flatten
    return normalized.flatten()

def detect_and_crop_face(image_path=None, image_array=None):
    """Deteksi wajah dengan Haar Cascade, return crop wajah."""
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = image_array
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        # Fallback: pakai seluruh gambar jika wajah tidak terdeteksi
        return load_and_preprocess(image_array=img), False
    
    x, y, w, h = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return face_normalized.flatten(), True