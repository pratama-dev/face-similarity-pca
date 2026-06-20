"""
eigenfaces_utils.py
====================
Modul utilitas untuk Verifikasi Wajah dengan Eigenfaces (PCA/SVD).

Seluruh logika di file ini diadaptasi LANGSUNG dari notebook
'Eigenfaces_Training_Verifikasi_Wajah.ipynb' (deteksi wajah dengan Haar
Cascade, preprocessing 100x100, training PCA dari foto masa kecil, lalu
proyeksi + skor kemiripan untuk foto sekarang). Bagian upload Google Colab
(`google.colab.files.upload()`) diganti dengan `st.file_uploader` di app.py,
tapi pipeline pengolahan gambar & matematikanya identik dengan notebook.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, wajib untuk server Streamlit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams["figure.facecolor"] = "white"


# ---------------------------------------------------------------------------
# 1) Deteksi Wajah & Preprocessing  (-> notebook Cell 4)
# ---------------------------------------------------------------------------

def get_face_cascade():
    """Load Haar Cascade bawaan OpenCV untuk deteksi wajah frontal."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_image_from_bytes(image_bytes):
    """Decode bytes hasil upload menjadi array gambar OpenCV (BGR)."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img_bgr


def detect_and_crop_face(img_bgr, cascade):
    """
    Deteksi wajah dengan Haar Cascade pada versi grayscale gambar.
    - Wajah terdeteksi -> crop area wajah terbesar.
    - Tidak terdeteksi  -> fallback memakai full image (grayscale).
    Mengembalikan: (crop_grayscale, status_terdeteksi)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = gray[y:y + h, x:x + w]
        terdeteksi = True
    else:
        crop = gray
        terdeteksi = False

    return crop, terdeteksi


def preprocess_face(face_crop, size=(100, 100)):
    """
    - Resize ke 100x100
    - Normalisasi piksel ke rentang 0-1
    - Flatten menjadi vektor 10.000 dimensi
    """
    resized = cv2.resize(face_crop, size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float64) / 255.0
    flattened = normalized.flatten()
    return resized, normalized, flattened


def proses_satu_foto(filename, image_bytes, cascade):
    """Pipeline lengkap: load -> deteksi wajah -> crop -> preprocessing."""
    img_bgr = load_image_from_bytes(image_bytes)
    if img_bgr is None:
        raise ValueError(
            f"Gagal membaca gambar '{filename}'. Pastikan file adalah gambar yang valid (jpg/jpeg/png)."
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_crop, terdeteksi = detect_and_crop_face(img_bgr, cascade)
    resized, normalized, flattened = preprocess_face(face_crop)

    return {
        "filename": filename,
        "original_rgb": img_rgb,
        "face_terdeteksi": terdeteksi,
        "preprocessed": resized,
        "normalized": normalized,
        "vector": flattened,
    }


def buat_grid_foto(daftar_data, judul, max_kolom=5):
    """Buat figure grid foto hasil preprocessing (jumlah foto fleksibel). (-> notebook Cell 4 & 10)"""
    n = len(daftar_data)
    kolom = min(max_kolom, n)
    baris = int(np.ceil(n / kolom))

    fig, axes = plt.subplots(baris, kolom, figsize=(3 * kolom, 3 * baris))
    axes = np.atleast_1d(axes).flatten()

    for ax, d in zip(axes, daftar_data):
        ax.imshow(d["preprocessed"], cmap="gray")
        ket = "✅ wajah" if d["face_terdeteksi"] else "⚠️ fallback"
        ax.set_title(f"{d['label']}\n{ket}", fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle(judul, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2) Training Model: Mean Face, Centering, PCA/SVD  (-> notebook Cell 12 & 16)
# ---------------------------------------------------------------------------

def latih_model(data_training):
    """
    Training model ruang wajah (Eigenfaces) HANYA dari foto-foto training.
    Foto sekarang TIDAK dilibatkan di tahap ini.
    Mengembalikan dict berisi semua komponen model yang sudah dilatih.
    """
    n_foto_training = len(data_training)
    X_train = np.array([d["vector"] for d in data_training])

    # Mean face HANYA dari data training
    mean_face_vector = X_train.mean(axis=0)
    mean_face_img = mean_face_vector.reshape(100, 100)

    # Centering
    X_train_centered = X_train - mean_face_vector

    # PCA/SVD -> eigenfaces
    n_components = max(1, n_foto_training - 1)
    pca = PCA(n_components=n_components, svd_solver="full")
    X_train_proj = pca.fit_transform(X_train_centered)

    # Centroid & skala variasi alami training (di eigenspace)
    centroid = X_train_proj.mean(axis=0)
    intra_dists = np.linalg.norm(X_train_proj - centroid, axis=1)
    skala_variasi = np.mean(intra_dists) if np.mean(intra_dists) > 0 else 1e-9

    return {
        "n_foto_training": n_foto_training,
        "X_train": X_train,
        "mean_face_vector": mean_face_vector,
        "mean_face_img": mean_face_img,
        "n_components": n_components,
        "pca": pca,
        "X_train_proj": X_train_proj,
        "centroid": centroid,
        "intra_dists": intra_dists,
        "skala_variasi": skala_variasi,
    }


def buat_figure_eigenfaces(model, max_tampil=5):
    """Visualisasi mean face & eigenfaces hasil training. (-> notebook Cell 14)"""
    n_tampil = min(max_tampil, model["n_components"])
    fig, axes = plt.subplots(1, n_tampil + 1, figsize=(4 * (n_tampil + 1), 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(model["mean_face_img"], cmap="gray")
    axes[0].set_title("Mean Face\n(training)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    pca = model["pca"]
    for i in range(n_tampil):
        eigenface = pca.components_[i].reshape(100, 100)
        axes[i + 1].imshow(eigenface, cmap="gray")
        axes[i + 1].set_title(
            f"Eigenface {i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=11
        )
        axes[i + 1].axis("off")

    plt.suptitle(
        f"Mean Face & Eigenfaces (dari {model['n_foto_training']} foto training, "
        f"total {model['n_components']} komponen)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3) Verifikasi Foto Sekarang  (-> notebook Cell 20 & 22)
# ---------------------------------------------------------------------------

def buat_figure_perbandingan(mean_face_img, data_sekarang):
    """Visualisasi: foto sekarang vs mean face training."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_face_img, cmap="gray")
    axes[0].set_title("Mean Face\n(hasil training)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(data_sekarang["preprocessed"], cmap="gray")
    ket = "✅ wajah" if data_sekarang["face_terdeteksi"] else "⚠️ fallback"
    axes[1].set_title(f"Foto Sekarang\n{ket}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def verifikasi_foto(model, data_sekarang):
    """
    Proyeksikan foto sekarang ke model terlatih (pca.transform, BUKAN
    pca.fit_transform — model tidak dilatih ulang), lalu hitung skor
    kemiripan terhadap centroid training.
    """
    pca = model["pca"]
    mean_face_vector = model["mean_face_vector"]
    centroid = model["centroid"]
    skala_variasi = model["skala_variasi"]

    current_vector = data_sekarang["vector"]
    current_centered = current_vector - mean_face_vector
    current_proj = pca.transform([current_centered])[0]

    # Cosine similarity ke centroid training -> [-1, 1] dipetakan ke [0, 100]%
    cos_sim_raw = cosine_similarity([current_proj], [centroid])[0][0]
    cos_pct = ((cos_sim_raw + 1) / 2) * 100

    # Euclidean distance ke centroid, dipetakan relatif terhadap variasi alami training
    current_dist = np.linalg.norm(current_proj - centroid)
    eucl_pct = 100 / (1 + (current_dist / skala_variasi))

    # Skor akhir = rata-rata cosine% dan euclidean%
    skor_akhir = (cos_pct + eucl_pct) / 2

    return {
        "current_proj": current_proj,
        "cos_sim_raw": cos_sim_raw,
        "cos_pct": cos_pct,
        "current_dist": current_dist,
        "eucl_pct": eucl_pct,
        "skor_akhir": skor_akhir,
    }


# ---------------------------------------------------------------------------
# 4) Hasil Akhir & Visualisasi  (-> notebook Cell 24 & 26)
# ---------------------------------------------------------------------------

def buat_figure_hasil(skor_akhir, threshold_pct=70):
    """Visualisasi bar chart hasil persentase kemiripan."""
    mirip = skor_akhir >= threshold_pct
    fig, ax = plt.subplots(figsize=(7, 2.3))
    warna = "#2E7D32" if mirip else "#E53935"
    ax.barh(["Kemiripan"], [skor_akhir], color=warna, height=0.5)
    ax.axvline(
        x=threshold_pct, color="red", linestyle="--", linewidth=1.5,
        label=f"Threshold MIRIP ({threshold_pct:.0f}%)",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Persentase (%)")
    ax.set_title(
        f"Kemiripan Foto Sekarang dengan Model Wajah Masa Kecil: {skor_akhir:.1f}%",
        fontsize=12, fontweight="bold",
    )
    label_x = min(skor_akhir + 1.5, 96)
    ax.text(label_x, 0, f"{skor_akhir:.1f}%", va="center", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
