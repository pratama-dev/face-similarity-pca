import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from eigenfaces_utils import (
    buat_figure_eigenfaces,
    buat_figure_hasil,
    buat_figure_perbandingan,
    buat_grid_foto,
    get_face_cascade,
    latih_model,
    proses_satu_foto,
    verifikasi_foto,
)

# ===========================================================================
# 1. Konfigurasi Halaman & Inject CSS Kustom (Perbaikan Kontras & Animasi)
# ===========================================================================
st.set_page_config(
    page_title="FaceVerify Pro - Eigenfaces",
    layout="wide",
    initial_sidebar_state="expanded",
)

def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #f9fafb;
        }

        /* Mewarnai Sidebar Tanpa Elemen AI Generatif */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: #f8fafc !important;
        }

        /* Desain Judul Utama Modern */
        .main-title {
            font-weight: 700;
            color: #0f172a;
            font-size: 2.5rem !important;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        .sub-title {
            color: #64748b;
            font-size: 1.1rem;
            margin-top: 0px;
            margin-bottom: 32px;
        }

        /* Langkah Badge Pengganti Emoticon */
        .step-badge {
            display: inline-block;
            background-color: #e0e7ff;
            color: #4f46e5 !important;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 9999px;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Gaya Kartu Utama - Memaksa Warna Teks Gelap Agar Selalu Terbaca */
        .custom-card {
            background-color: #ffffff !important;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 32px;
            border: 1px solid #e2e8f0;
        }
        
        /* Mengatasi masalah tulisan putih di dark mode */
        .custom-card, 
        .custom-card p, 
        .custom-card span, 
        .custom-card sm, 
        .custom-card div,
        .custom-card h1, 
        .custom-card h2, 
        .custom-card h3 {
            color: #1e293b !important;
        }
        
        .card-header {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a !important;
            margin-bottom: 12px;
        }

        /* Area Upload */
        [data-testid="stFileUploadDropzone"] {
            background-color: #f8fafc !important;
            border: 2px dashed #cbd5e1 !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploadDropzone"] * {
            color: #334155 !important;
        }

        /* Tombol Interaktif dengan Animasi Mekanik */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
            color: #ffffff !important;
            font-weight: 600;
            font-size: 1rem;
            padding: 14px 28px;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
            margin-top: 16px;
        }

        /* Efek Hover */
        div.stButton > button:first-child:hover {
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
            transform: translateY(-2px);
            background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
            border: none;
        }

        /* Efek Tekan Kebawah Sesuai Permintaan */
        div.stButton > button:first-child:active {
            transform: translateY(2px) scale(0.98);
            box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2);
            background: linear-gradient(135deg, #3730a3 0%, #2e2685 100%);
            transition: all 0.05s ease;
        }

        /* Tabel Data Hasil Analisis */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            margin-bottom: 20px;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        .data-table th {
            background-color: #f1f5f9;
            color: #475569 !important;
            font-weight: 600;
            text-align: left;
            padding: 14px;
            font-size: 0.9rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .data-table td {
            padding: 14px;
            color: #334155 !important;
            font-size: 0.9rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .data-table tr:last-child td {
            border-bottom: none;
        }

        /* Komponen Animasi Berjalan Bilah Kemiripan */
        .similarity-container {
            background-color: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            border: 1px solid #e2e8f0;
        }
        .similarity-label {
            font-size: 0.9rem;
            color: #475569 !important;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .progress-bg {
            background-color: #e2e8f0;
            border-radius: 9999px;
            height: 20px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #4f46e5, #10b981);
            height: 100%;
            border-radius: 9999px;
            width: 0%;
            animation: jalanProgress 1.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        @keyframes jalanProgress {
            to { width: var(--target-width); }
        }
        .similarity-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a !important;
            margin-top: 10px;
            text-align: right;
        }

        /* Kartu Hasil Akhir */
        .result-card-success {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 24px;
            border-radius: 12px;
            border-left: 6px solid #16a34a;
            color: #14532d !important;
            margin-top: 24px;
            animation: munculHasil 0.4s ease-out;
        }
        .result-card-fail {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            padding: 24px;
            border-radius: 12px;
            border-left: 6px solid #dc2626;
            color: #7f1d1d !important;
            margin-top: 24px;
            animation: munculHasil 0.4s ease-out;
        }
        @keyframes munculHasil {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ===========================================================================
# 2. Logika Utama Sistem
# ===========================================================================

@st.cache_resource
def load_cascade():
    return get_face_cascade()

face_cascade = load_cascade()

for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar Kontrol Sistem
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Informasi Sistem")
    st.markdown(
        """
        Aplikasi ini dianalisis menggunakan metode perhitungan matematis 
        Eigenfaces berbasis PCA dan SVD untuk menguji struktur kemiripan wajah.
        
        Sistem bekerja dengan memproyeksikan data gambar baru ke dalam ruang 
        vektor yang telah dibentuk oleh kumpulan data latihan.
        """, unsafe_allow_html=True
    )

    threshold_pct = st.slider(
        "Ambang Batas Verifikasi (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=1,
        help="Skor kumulatif yang harus terpenuhi untuk dinyatakan sebagai orang yang sama.",
    )

    if st.session_state.model is not None:
        m = st.session_state.model
        st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color: rgba(255,255,255,0.1); padding: 16px; border-radius: 8px;">
                <span style="font-weight:600; display:block; margin-bottom:4px;">Model Aktif</span>
                <small style="display:block;">Data Latihan: {m['n_foto_training']} Berkas</small>
                <small style="display:block;">Komponen Utama: {m['n_components']} Vektor</small>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Basis Data"):
            for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi"]:
                st.session_state[key] = None
            st.rerun()

# ---------------------------------------------------------------------------
# Header Utama Aplikasi
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-title">Sistem Analisis Verifikasi Wajah</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ekstraksi Komponen Wajah Berbasis Ruang Vektor Eigenfaces</p>', unsafe_allow_html=True)

# ===========================================================================
# TAHAP 1: PENGUMPULAN DATA LATIHAN
# ===========================================================================
st.markdown(
    """
    <div class="custom-card">
        <div class="step-badge">Tahap 01</div>
        <div class="card-header">Pendaftaran Struktur Wajah Masa Kecil</div>
        <p style="margin-bottom: 20px;">Silakan unggah minimal dua buah dokumen foto masa kecil Anda. Sistem akan melakukan 
        normalisasi gambar untuk mengenali karakteristik dasar geometri wajah Anda secara otomatis.</p>
    </div>
    """, unsafe_allow_html=True
)

# Menempatkan uploader dan tombol di luar elemen HTML agar fungsi Streamlit berjalan normal
uploaded_training = st.file_uploader(
    "Pilih dokumen foto masa kecil",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="uploader_training",
    label_visibility="collapsed"
)

train_btn = st.button("Ekstrak Fitur Wajah", disabled=not uploaded_training)

if train_btn:
    if len(uploaded_training) < 2:
        st.error("Sistem memerlukan minimal dua berkas foto latihan untuk melakukan kalkulasi matriks ragam.")
    else:
        with st.spinner("Sistem sedang mengekstraksi matriks komponen utama..."):
            data_training = []
            gagal = False
            for i, file in enumerate(uploaded_training):
                try:
                    hasil = proses_satu_foto(file.name, file.getvalue(), face_cascade)
                except ValueError as e:
                    st.error(str(e))
                    gagal = True
                    break
                hasil["label"] = f"Kecil {i + 1}"
                data_training.append(hasil)

            if not gagal:
                model = latih_model(data_training)
                st.session_state.data_training = data_training
                st.session_state.model = model
                st.session_state.data_sekarang = None
                st.session_state.hasil_verifikasi = None
                st.rerun()

# Tampilan Visualisasi Hasil Ekstraksi Tahap 1
if st.session_state.data_training is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Hasil Normalisasi Citra Digital (Skala Abu-abu 100x100)")
    st.pyplot(buat_grid_foto(st.session_state.data_training, ""))
    
    st.subheader("Rata-rata Geometri dan Komponen Utama Wajah (Eigenfaces)")
    st.pyplot(buat_figure_eigenfaces(st.session_state.model))

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ===========================================================================
# TAHAP 2: PROSES UJI VERIFIKASI
# ===========================================================================
st.markdown(
    """
    <div class="custom-card">
        <div class="step-badge">Tahap 02</div>
        <div class="card-header">Uji Banding Dokumen Wajah Sekarang</div>
        <p style="margin-bottom: 20px;">Unggah satu buah foto kondisi terbaru Anda untuk diproyeksikan langsung ke dalam 
        model ruang vektor masa kecil yang telah terbentuk sebelumnya.</p>
    </div>
    """, unsafe_allow_html=True
)

if st.session_state.model is None:
    st.info("Silakan selesaikan Tahap 01 terlebih dahulu untuk mengaktifkan fungsi uji banding.")
else:
    uploaded_sekarang = st.file_uploader(
        "Pilih dokumen foto sekarang",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="uploader_sekarang",
        label_visibility="collapsed"
    )

    verify_btn = st.button("Jalankan Komparasi Wajah", disabled=uploaded_sekarang is None)

    if verify_btn:
        with st.spinner("Menghitung koefisien proyeksi jarak euclidean..."):
            try:
                data_sekarang = proses_satu_foto(
                    uploaded_sekarang.name, uploaded_sekarang.getvalue(), face_cascade
                )
                data_sekarang["label"] = "Foto Sekarang"
                hasil_verifikasi = verifikasi_foto(st.session_state.model, data_sekarang)
                st.session_state.data_sekarang = data_sekarang
                st.session_state.hasil_verifikasi = hasil_verifikasi
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # Tampilan Output Analisis Tahap 2
    if st.session_state.data_sekarang is not None and st.session_state.hasil_verifikasi is not None:
        model = st.session_state.model
        data_sekarang = st.session_state.data_sekarang
        hasil_verifikasi = st.session_state.hasil_verifikasi
        skor_akhir = hasil_verifikasi["skor_akhir"]
        mirip = (skor_akhir / 100) >= (threshold_pct / 100)
        
        # Otomatis geser layar fokus ke hasil analisis
        components.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 1200);</script>", height=0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Komparasi Visual Terhadap Wajah Rata-rata")
        st.pyplot(buat_figure_perbandingan(model["mean_face_img"], data_sekarang))

        # Komponen Animasi Berjalan Bilah Kemiripan
        st.markdown(f"""
            <div class="similarity-container">
                <div class="similarity-label">Bilah Kemiripan Dinamis</div>
                <div class="progress-bg">
                    <div class="progress-bar-fill" style="--target-width: {skor_akhir}%;"></div>
                </div>
                <div class="similarity-value">{skor_akhir:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        # Tabel Pengolahan Data Hasil Face Similarity
        st.markdown("### Parameter Hasil Analisis Metrik Wajah")
        st.markdown(f"""
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Parameter Uji Geometri</th>
                        <th>Nilai Perhitungan Kelompok</th>
                        <th>Ambang Batas Pengujian</th>
                        <th>Status Evaluasi Teknis</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Koefisien Cosine Similarity</td>
                        <td>{hasil_verifikasi['cos_pct']:.2f}%</td>
                        <td>{threshold_pct:.0f}%</td>
                        <td>{"Terpenuhi" if hasil_verifikasi['cos_pct'] >= threshold_pct else "Kurang"}</td>
                    </tr>
                    <tr>
                        <td>Jarak Relatif Euclidean Distance</td>
                        <td>{hasil_verifikasi['eucl_pct']:.2f}%</td>
                        <td>{threshold_pct:.0f}%</td>
                        <td>{"Terpenuhi" if hasil_verifikasi['eucl_pct'] >= threshold_pct else "Kurang"}</td>
                    </tr>
                    <tr>
                        <td><strong>Akumulasi Skor Akhir Kombinasi</strong></td>
                        <td><strong>{skor_akhir:.2f}%</strong></td>
                        <td><strong>{threshold_pct:.0f}%</strong></td>
                        <td><strong>{"Lolos Evaluasi" if mirip else "Ditolak"}</strong></td>
                    </tr>
                </tbody>
            </table>
        """, unsafe_allow_html=True)

        st.subheader("Grafik Posisi Distribusi Ambang Batas")
        st.pyplot(buat_figure_hasil(skor_akhir, threshold_pct))

        # Keputusan Akhir Sistem Tanpa Emoticon
        if mirip:
            st.markdown(f"""
                <div class="result-card-success">
                    <h3 style="margin-top:0;">HASIL EVALUASI: TERVERIFIKASI</h3>
                    <p>Sistem mendeteksi bahwa pola struktur komponen utama pada wajah sekarang 
                    memiliki konsistensi matematis yang signifikan dengan data masa kecil Anda.</p>
                    <strong>Status: Lolos Verifikasi Identitas Diri</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-fail">
                    <h3 style="margin-top:0;">HASIL EVALUASI: TIDAK TERVERIFIKASI</h3>
                    <p>Sistem menemukan deviasi spasial yang terlalu besar antara proyeksi foto wajah sekarang 
                    terhadap matriks ruang vektor masa kecil Anda.</p>
                    <strong>Status: Gagal Verifikasi Identitas Diri</strong>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748b; padding: 24px; background-color: #f1f5f9; border-radius: 12px; border: 1px solid #e2e8f0;">
        <span style="font-weight: 600; display: block; margin-bottom: 4px; color: #334155;">Dokumentasi Metode Analisis</span>
        Aplikasi ini menggunakan teknik reduksi dimensi Principal Component Analysis atau Singular Value Decomposition. 
        Akurasi perhitungan sangat dipengaruhi oleh variasi sudut pengambilan gambar, intensitas cahaya, serta ekspresi wajah pada objek foto latihan.
    </div>
    <br>
    """, unsafe_allow_html=True
)
