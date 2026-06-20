import numpy as np
import streamlit as st
import streamlit.components.v1 as components # Untuk trik animasi/scroll
import base64 # Untuk background image jika perlu, atau embed asset

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
# 1. Konfigurasi Halaman & Inject CSS Kustom (The Magic Sauce)
# ===========================================================================
st.set_page_config(
    page_title="FaceVerify Pro - Eigenfaces",
    page_icon="🧑‍🦱",
    layout="wide",
    initial_sidebar_state="expanded",
)

def local_css():
    st.markdown("""
    <style>
        /* Impor Font Modern */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        /* Terapkan font ke seluruh app */
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* Background Halaman */
        .stApp {
            background-color: #f8faff;
        }

        /* Mengubah gaya Sidebar */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #2e3192 0%, #1bffff 100%);
            color: white;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: white !important;
        }
        /* Slider Kustom di Sidebar */
        .stSlider [data-baseweb="slider"] {
            margin-bottom: 25px;
        }

        /* Mengubah Gaya Judul Utama */
        .main-title {
            font-weight: 700;
            color: #1e3a8a;
            font-size: 3rem !important;
            margin-bottom: 0px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-title {
            color: #6b7280;
            font-size: 1.2rem;
            margin-top: 0px;
            margin-bottom: 30px;
        }

        /* -- Gaya KARTU (Card Layout) -- */
        .custom-card {
            background-color: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            margin-bottom: 30px;
            border: 1px solid #e5e7eb;
            transition: transform 0.3s ease;
        }
        .custom-card:hover {
            transform: translateY(-5px);
        }
        .card-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Gaya Khusus Area Upload */
        [data-testid="stFileUploadDropzone"] {
            background-color: #eff6ff;
            border: 2px dashed #3b82f6;
            border-radius: 15px;
        }

        /* -- ANIMASI & GAYA TOMBOL UTAMA -- */
        div.stButton > button:first-child {
            background: linear-gradient(45deg, #ff00cc, #333399);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 12px 30px;
            border-radius: 50px;
            border: none;
            box-shadow: 0 4px 15px rgba(51, 51, 153, 0.4);
            
            /* Transisi Halus untuk semua properti */
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            width: 100%;
            margin-top: 10px;
        }

        /* Efek Hover (Mouse di atas) */
        div.stButton > button:first-child:hover {
            box-shadow: 0 6px 20px rgba(51, 51, 153, 0.6);
            transform: translateY(-2px);
            background: linear-gradient(45deg, #ff00cc, #4a4ae6); /* Sedikit lebih terang */
            border: none;
            color: white;
        }

        /* Efek Active (Saat Dipencet) - Ini yang diminta user */
        div.stButton > button:first-child:active {
            transform: translateY(2px) scale(0.98); /* Terlihat mendem */
            box-shadow: 0 2px 5px rgba(51, 51, 153, 0.5);
            background: linear-gradient(45deg, #d400ab, #28287a); /* Sedikit lebih gelap */
            transition: all 0.1s ease; /* Respon cepat saat klik */
        }
        
        /* Tombol Reset di Sidebar */
        div.stButton > button[kind="secondary"] {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid white;
            border-radius: 10px;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: white;
            color: #2e3192;
        }

        /* -- Gaya Metrik (Hasil Angka) -- */
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #333399;
        }

        /* -- Kartu Hasil Verifikasi (SUCCESS/ERROR Kustom) -- */
        .result-card-success {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 10px solid #22863a;
            color: #155724;
            margin-top: 20px;
            animation: fadeIn 0.5s;
        }
        .result-card-fail {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            padding: 25px;
            border-radius: 15px;
            border-left: 10px solid #cb2431;
            color: #721c24;
            margin-top: 20px;
            animation: fadeIn 0.5s;
        }
        
        /* Animasi Fade In sederhana */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

    </style>
    """, unsafe_allow_html=True)

local_css()

# ===========================================================================
# 2. Logika Utama (Tetap dipertahankan)
# ===========================================================================

@st.cache_resource
def load_cascade():
    return get_face_cascade()

face_cascade = load_cascade()

# Inisialisasi session state
for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi", "training_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar (Dipercantik)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ FaceVerify Pro")
    st.markdown(
        """
        Aplikasi ini menggunakan teknologi **Eigenfaces (PCA/SVD)** untuk membandingkan 
        apakah wajah Anda sekarang masih mengenali wajah Anda di masa kecil.

        **Cara Pakai:**
        1. 📸 Upload foto-foto masa kecil (min. 2)
        2. ⚙️ Latih AI
        3. 🔍 Upload foto sekarang & verifikasi

        ---
        <small>⚠️ *Hanya untuk edukasi, bukan verifikasi keamanan hukum.*</small>
        """, unsafe_allow_html=True
    )

    threshold_pct = st.slider(
        "🎚️ Sensitivitas 'MIRIP' (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=1,
        help="Semakin tinggi, semakin sulit dianggap mirip.",
    )

    if st.session_state.model is not None:
        m = st.session_state.model
        st.markdown("---")
        st.success(
            f"✅ **AI Terlatih**\n"
            f"- {m['n_foto_training']} Foto Kecil\n"
            f"- {m['n_components']} Eigenfaces"
        )
        if st.button("🔄 Reset Sistem"):
            for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi", "training_done"]:
                st.session_state[key] = None
            st.rerun()

# ---------------------------------------------------------------------------
# Konten Utama
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-title">Wajah Masa Kecil vs Sekarang</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Mari buktikan dengan Algoritma Eigenfaces (PCA)</p>', unsafe_allow_html=True)

# Layout Kolom Besar (1. Training, 2. Verification)
col_train, col_verify = st.columns([1, 1], gap="large")

# ===========================================================================
# LANGKAH 1: TRAINING (Kolom Kiri)
# ===========================================================================
with col_train:
    # Membungkus dalam kartu kustom
    st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span>1️⃣</span> Latih Memori AI (Foto Masa Kecil)
            </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        "Upload minimal **2 foto** masa kecilmu. AI akan mempelajari "
        "pola wajah dasar kamu."
    )

    uploaded_training = st.file_uploader(
        "Pilih foto-foto masa kecil...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="uploader_training",
    )

    # Styling tombol dipencet ditangani CSS (div.stButton...)
    train_btn = st.button("🚀 Mulai Latih AI", disabled=not uploaded_training)

    if train_btn:
        if len(uploaded_training) < 2:
            st.error("⚠️ Butuh minimal 2 foto untuk menghitung PCA.")
        else:
            with st.spinner("🧠 AI sedang mempelajari wajah masa kecilmu..."):
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
                    st.session_state.training_done = True
                    st.success("✅ Proses Latihan Selesai!")

    # Penutup Kartu
    st.markdown("</div>", unsafe_allow_html=True)

    # Hasil Training (Tampilkan di bawah kartu jika ada)
    if st.session_state.data_training is not None:
        with st.expander("📋 Lihat Hasil Pemrosesan Wajah (Grayscale)", expanded=True):
            data_training = st.session_state.data_training
            model = st.session_state.model

            st.pyplot(buat_grid_foto(data_training, ""))
            
            st.markdown("---")
            st.markdown("**'Wajah Rata-rata' & Fitur Utama (Eigenfaces)**")
            st.pyplot(buat_figure_eigenfaces(model))
            
            with st.expander("🔬 Detail Teknis Data", expanded=False):
                st.write(f"Variansi Total: {np.sum(model['pca'].explained_variance_ratio_) * 100:.2f}%")
                st.write(f"Skala Variasi Alami: {model['skala_variasi']:.4f}")


# ===========================================================================
# LANGKAH 2: VERIFIKASI (Kolom Kanan)
# ===========================================================================
with col_verify:
    st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span>2️⃣</span> Uji Verifikasi (Foto Sekarang)
            </div>
    """, unsafe_allow_html=True)

    if st.session_state.model is None:
        st.info("👈 Selesaikan Langkah 1 dulu untuk melatih AI.")
    else:
        uploaded_sekarang = st.file_uploader(
            "Upload 1 foto Anda sekarang...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploader_sekarang",
        )

        verify_btn = st.button("🔍 Verifikasi Wajah", disabled=uploaded_sekarang is None)

        if verify_btn:
            with st.spinner("🕵️ Membandingkan dengan memori masa kecil..."):
                try:
                    data_sekarang = proses_satu_foto(
                        uploaded_sekarang.name, uploaded_sekarang.getvalue(), face_cascade
                    )
                    data_sekarang["label"] = "Foto Sekarang"
                    hasil_verifikasi = verifikasi_foto(st.session_state.model, data_sekarang)
                    st.session_state.data_sekarang = data_sekarang
                    st.session_state.hasil_verifikasi = hasil_verifikasi
                except ValueError as e:
                    st.error(str(e))

    # Penutup Kartu
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Area Hasil Verifikasi (Colorful & Animated)
    # -----------------------------------------------------------------------
    if st.session_state.data_sekarang is not None and st.session_state.hasil_verifikasi is not None:
        model = st.session_state.model
        data_sekarang = st.session_state.data_sekarang
        hasil_verifikasi = st.session_state.hasil_verifikasi
        skor_akhir = hasil_verifikasi["skor_akhir"]
        
        # Trik scroll otomatis ke hasil
        components.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 500);</script>", height=0)

        st.markdown("### 📊 Perbandingan & Skor")
        
        # Plot Perbandingan
        fig_compare = buat_figure_perbandingan(model["mean_face_img"], data_sekarang)
        st.pyplot(fig_compare)

        # Metrik Berwarna
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Skor Kemiripan Akhir", f"{skor_akhir:.2f}%")
        c2.metric("Threshold Dibutuhkan", f"{threshold_pct:.0f}%")

        # Visualisasi Gauge
        fig_hasil = buat_figure_hasil(skor_akhir, threshold_pct)
        st.pyplot(fig_hasil)

        # KESIMPULAN DENGAN KARTU KUSTOM
        mirip = (skor_akhir / 100) >= (threshold_pct / 100)
        
        if mirip:
            st.markdown(f"""
                <div class="result-card-success">
                    <h3 style="color: #155724; margin-top:0;"> HASIL: MIRIP!</h3>
                    <p>AI mengenali pola wajah masa kecil Anda pada foto sekarang. 
                    Anda masih terlihat seperti diri Anda yang dulu (konsisten).</p>
                    <strong style="font-size: 1.2rem;">Skor: {skor_akhir:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-fail">
                    <h3 style="color: #721c24; margin-top:0;"> HASIL: TIDAK MIRIP</h3>
                    <p>AI menilai pola wajah sekarang terlalu berbeda dengan data latihan masa kecil. 
                    (Faktor usia, sudut, atau pencahayaan mungkin berpengaruh).</p>
                    <strong style="font-size: 1.2rem;">Skor: {skor_akhir:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; padding: 20px; background-color: #f1f5f9; border-radius: 10px;">
        <h4> Catatan Edukasi</h4>
        Metode Eigenfaces menggunakan analisis statistik (PCA) untuk mencari komponen utama wajah. 
        Makin banyak & beragam foto training, makin akurat model mengenali variasi wajah Anda.
    </div>
    <br>
    """, unsafe_allow_html=True
)
