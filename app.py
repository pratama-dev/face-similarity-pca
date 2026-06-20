"""
app.py
=======
Streamlit App: Verifikasi Wajah dengan Eigenfaces (PCA/SVD)

Diadaptasi dari notebook 'Eigenfaces_Training_Verifikasi_Wajah.ipynb'.
Karena Streamlit berjalan di server (bukan Colab), upload foto memakai
st.file_uploader sebagai pengganti google.colab.files.upload(), tapi
seluruh pipeline pengolahan gambar & perhitungan PCA/kemiripan identik
dengan notebook aslinya (lihat eigenfaces_utils.py).

Alur:
1. Upload & latih model dari beberapa foto masa kecil (training)
2. Upload 1 foto sekarang (query) -> diproyeksikan ke model -> skor kemiripan

Cara menjalankan:
    streamlit run app.py
"""

import numpy as np
import streamlit as st

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

st.set_page_config(
    page_title="Verifikasi Wajah - Eigenfaces",
    page_icon="🧑‍🦱",
    layout="wide",
)


@st.cache_resource
def load_cascade():
    return get_face_cascade()


face_cascade = load_cascade()

# ---------------------------------------------------------------------------
# Inisialisasi session state
# ---------------------------------------------------------------------------
for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("ℹ️ Tentang Aplikasi")
st.sidebar.markdown(
    """
Aplikasi ini mendemonstrasikan **Eigenfaces (PCA/SVD)** untuk verifikasi
wajah sederhana, diadaptasi dari notebook training & verifikasi wajah.

**Alur kerja:**
1. Upload beberapa foto masa kecil → latih model
2. Upload 1 foto sekarang → verifikasi terhadap model

⚠️ *Tujuan edukasi mempelajari PCA/SVD, bukan alat verifikasi identitas
forensik/hukum.*
"""
)

threshold_pct = st.sidebar.slider(
    "Threshold 'MIRIP' (%)",
    min_value=50,
    max_value=95,
    value=70,
    step=1,
    help="Skor kemiripan >= nilai ini dianggap MIRIP (default notebook: 70%).",
)

if st.session_state.model is not None:
    m = st.session_state.model
    st.sidebar.success(
        f"✅ Model terlatih dari {m['n_foto_training']} foto "
        f"({m['n_components']} eigenfaces)."
    )
    if st.sidebar.button("🔄 Reset Model"):
        for key in ["model", "data_training", "data_sekarang", "hasil_verifikasi"]:
            st.session_state[key] = None
        st.rerun()

st.title('🧑\u200d🦱➡️🧔 Verifikasi Wajah: "Apakah Foto Sekarang Ini Benar-Benar Aku?"')
st.caption("Metode: PCA / SVD (Eigenfaces) — Training & Verifikasi")

# ---------------------------------------------------------------------------
# Langkah 1: Upload & Training
# ---------------------------------------------------------------------------
st.header("1️⃣ Upload Foto Masa Kecil (Data Training)")
st.markdown(
    "Upload **sebebas mungkin** foto masa kecil kamu — **minimal 2 foto** "
    "(disarankan 3+ supaya model belajar variasi wajahmu dari berbagai sudut/ekspresi/usia)."
)

uploaded_training = st.file_uploader(
    "Upload foto masa kecil (boleh pilih banyak file sekaligus)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="uploader_training",
)

if st.button("🚀 Proses & Latih Model", disabled=not uploaded_training):
    if len(uploaded_training) < 2:
        st.error(
            "Minimal butuh 2 foto training agar PCA bisa dihitung. "
            "Silakan upload lebih banyak foto."
        )
    else:
        if len(uploaded_training) < 3:
            st.warning(
                f"Kamu upload {len(uploaded_training)} foto. Disarankan minimal 3 foto "
                "agar model lebih akurat, tapi proses tetap dilanjutkan."
            )

        with st.spinner("Memproses foto & melatih model..."):
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
                # foto sekarang & hasil verifikasi sebelumnya jadi tidak relevan lagi
                st.session_state.data_sekarang = None
                st.session_state.hasil_verifikasi = None

        if not gagal:
            st.success(f"✅ {len(data_training)} foto training berhasil diproses & model dilatih.")

if st.session_state.data_training is not None:
    data_training = st.session_state.data_training
    model = st.session_state.model

    with st.expander("📋 Status deteksi wajah per foto", expanded=False):
        for d in data_training:
            status = "✅ wajah terdeteksi" if d["face_terdeteksi"] else "⚠️ fallback full image"
            st.write(f"**{d['label']}** ({d['filename']}): {status}")

    st.subheader("Foto Training Setelah Preprocessing (Grayscale 100×100)")
    fig_grid = buat_grid_foto(data_training, "Foto Masa Kecil Setelah Preprocessing")
    st.pyplot(fig_grid)

    st.subheader("Mean Face & Eigenfaces (Hasil Training)")
    fig_eigen = buat_figure_eigenfaces(model)
    st.pyplot(fig_eigen)

    with st.expander("📐 Detail Variansi Eigenfaces (untuk laporan)", expanded=False):
        pca = model["pca"]
        for i, variansi in enumerate(pca.explained_variance_ratio_):
            st.write(
                f"• Eigenface {i + 1} (Komponen Utama {i + 1}): menjelaskan sekitar "
                f"**{variansi * 100:.2f}%** variansi data."
            )
        st.write(
            f"**Total akumulasi ragam informasi:** "
            f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
        )

    with st.expander("🎯 Centroid & Variasi Alami Training", expanded=False):
        st.write(
            f"Rata-rata variasi alami antar foto training ke centroid: "
            f"**{model['skala_variasi']:.4f}**"
        )
        st.write("Jarak masing-masing foto training ke centroid:")
        st.write(np.round(model["intra_dists"], 3))

st.divider()

# ---------------------------------------------------------------------------
# Langkah 2: Upload & Verifikasi
# ---------------------------------------------------------------------------
st.header("2️⃣ Upload Foto Sekarang (Query) & Verifikasi")

if st.session_state.model is None:
    st.info("⬆️ Latih model dari foto masa kecil terlebih dahulu sebelum verifikasi.")
else:
    uploaded_sekarang = st.file_uploader(
        "Upload 1 foto sekarang",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="uploader_sekarang",
    )

    if st.button("🔍 Verifikasi", disabled=uploaded_sekarang is None):
        with st.spinner("Memproses & memproyeksikan foto ke model..."):
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

    if st.session_state.data_sekarang is not None and st.session_state.hasil_verifikasi is not None:
        model = st.session_state.model
        data_sekarang = st.session_state.data_sekarang
        hasil_verifikasi = st.session_state.hasil_verifikasi

        status = "✅ wajah terdeteksi" if data_sekarang["face_terdeteksi"] else "⚠️ fallback full image"
        st.write(f"**Status deteksi wajah:** {status}")

        st.subheader("Foto Sekarang vs Mean Face Training")
        fig_compare = buat_figure_perbandingan(model["mean_face_img"], data_sekarang)
        st.pyplot(fig_compare)

        st.subheader("Hasil Proyeksi & Kemiripan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cosine Similarity", f"{hasil_verifikasi['cos_pct']:.2f}%")
        col2.metric("Euclidean (relatif)", f"{hasil_verifikasi['eucl_pct']:.2f}%")
        col3.metric("Skor Akhir", f"{hasil_verifikasi['skor_akhir']:.2f}%")

        skor_akhir = hasil_verifikasi["skor_akhir"]
        mirip = (skor_akhir / 100) >= (threshold_pct / 100)
        kesimpulan = (
            "MIRIP ✅ (pola wajah sekarang konsisten dengan foto-foto masa kecil yang ditraining)"
            if mirip
            else "TIDAK MIRIP ❌ (pola wajah sekarang menyimpang dari foto-foto masa kecil yang ditraining)"
        )

        st.subheader("📊 Hasil Verifikasi")
        st.write(f"Jumlah foto training (masa kecil): **{model['n_foto_training']}**")
        st.write(f"Persentase kemiripan: **{skor_akhir:.2f}%**")
        st.write(f"Threshold: **{threshold_pct:.0f}%**")
        if mirip:
            st.success(f"Kesimpulan: {kesimpulan}")
        else:
            st.error(f"Kesimpulan: {kesimpulan}")

        st.subheader("Visualisasi Persentase Kemiripan")
        fig_hasil = buat_figure_hasil(skor_akhir, threshold_pct)
        st.pyplot(fig_hasil)

st.divider()
st.markdown(
    """
### 📝 Catatan Penting Tentang Hasil
- Aplikasi ini untuk **tujuan edukasi** mempelajari PCA/SVD (Eigenfaces) dalam skema
  training/verifikasi sederhana, **bukan** alat verifikasi identitas yang sahih secara
  forensik/hukum.
- Model ("centroid" + "skala variasi") dibangun **hanya dari foto training masa kecil**;
  foto sekarang cuma diproyeksikan ke model itu — ini meniru alur *train lalu test* yang benar.
- Makin banyak & makin beragam foto training (sudut, ekspresi, pencahayaan, rentang usia),
  makin baik estimasi kemiripannya.
"""
)
