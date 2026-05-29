import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import detect_and_crop_face

def load_dataset(dataset_path):
    """Load semua gambar dari folder dataset."""
    X, labels, paths = [], [], []
    
    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_folder, filename)
                try:
                    vector, _ = detect_and_crop_face(image_path=image_path)
                    X.append(vector)
                    labels.append(person_name)
                    paths.append(image_path)
                except:
                    continue
    
    return np.array(X), np.array(labels), paths

def train_pca(X, n_components=50):
    """Latih model PCA."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def compare_two_faces(vec1, vec2, pca):
    """Hitung cosine similarity dua wajah."""
    v1 = pca.transform(vec1.reshape(1, -1))
    v2 = pca.transform(vec2.reshape(1, -1))
    sim = cosine_similarity(v1, v2)[0][0]
    return float(sim)

def recognize_face(face_vector, pca, X_pca, labels, threshold=0.80):
    """Cari wajah paling mirip dari database."""
    face_pca = pca.transform(face_vector.reshape(1, -1))
    similarities = cosine_similarity(face_pca, X_pca)[0]
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    best_label = labels[best_idx]
    
    if best_sim >= threshold:
        return best_label, float(best_sim), similarities
    return "Tidak dikenal", float(best_sim), similarities

def save_model(pca, X_pca, labels, path="models/"):
    """Simpan model PCA."""
    os.makedirs(path, exist_ok=True)
    joblib.dump({'pca': pca, 'X_pca': X_pca, 'labels': labels}, 
                f"{path}pca_model.pkl")

def load_model(path="models/pca_model.pkl"):
    """Load model PCA."""
    data = joblib.load(path)
    return data['pca'], data['X_pca'], data['labels']