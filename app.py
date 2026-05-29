<<<<<<< HEAD
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import joblib

from preprocessing import detect_and_crop_face, load_and_preprocess
from pca_model import (load_dataset, train_pca, compare_two_faces,
                       recognize_face, save_model, load_model)
from eda import (plot_class_distribution, plot_explained_variance,
                 plot_eigenfaces, plot_pca_scatter, plot_mean_face)

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Face Similarity PCA",
    page_icon="👤",
    layout="wide"
)

MODEL_PATH = "models/pca_model.pkl"

# ===================== SIDEBAR =====================
st.sidebar.title("⚙️ Konfigurasi")
n_components = st.sidebar.slider("Jumlah Komponen PCA", 10, 150, 50)
threshold = st.sidebar.slider("Threshold Similarity", 0.5, 1.0, 0.80, 0.01)
dataset_path = st.sidebar.text_input("Path Dataset Train", "dataset/train")

# ===================== MAIN =====================
st.title("👤 Face Similarity Detection")
st.markdown("**Metode:** PCA/SVD (Eigenfaces) | **Framework:** Streamlit")

tabs = st.tabs(["🏋️ Training", "📊 EDA", 
                "🔍 Bandingkan 2 Wajah", "🎯 Identifikasi Wajah",
                "👶 Masa Kecil vs Dewasa"])

# ===================== TAB 1: TRAINING =====================
with tabs[0]:
    st.header("Training Model PCA")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Dataset path: `{dataset_path}`")
        if st.button("🚀 Mulai Training", type="primary"):
            with st.spinner("Memuat dataset..."):
                if not os.path.exists(dataset_path):
                    st.error("Folder dataset tidak ditemukan!")
                    st.stop()
                
                X, labels, paths = load_dataset(dataset_path)
                st.success(f"✅ Dataset dimuat: {len(X)} gambar, "
                           f"{len(set(labels))} orang")
            
            with st.spinner("Melatih PCA..."):
                pca, X_pca = train_pca(X, n_components=n_components)
                save_model(pca, X_pca, labels)
                
                st.session_state['pca'] = pca
                st.session_state['X_pca'] = X_pca
                st.session_state['labels'] = labels
                st.session_state['X_raw'] = X
                
                total_var = np.sum(pca.explained_variance_ratio_) * 100
                st.success(f"✅ PCA selesai! Total variance: {total_var:.1f}%")
    
    with col2:
        st.subheader("Info Model")
        if 'pca' in st.session_state:
            pca = st.session_state['pca']
            st.metric("Jumlah Komponen", n_components)
            st.metric("Total Gambar", len(st.session_state['labels']))
            st.metric("Total Orang", len(set(st.session_state['labels'])))
            st.metric("Explained Variance",
                      f"{np.sum(pca.explained_variance_ratio_)*100:.1f}%")
        else:
            # Coba load model tersimpan
            if os.path.exists(MODEL_PATH):
                pca, X_pca, labels = load_model(MODEL_PATH)
                st.session_state['pca'] = pca
                st.session_state['X_pca'] = X_pca
                st.session_state['labels'] = labels
                st.info("Model sebelumnya berhasil dimuat!")

# ===================== TAB 2: EDA =====================
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    
    if 'pca' not in st.session_state:
        st.warning("⚠️ Latih model terlebih dahulu di tab Training!")
    else:
        pca = st.session_state['pca']
        X_pca = st.session_state['X_pca']
        labels = st.session_state['labels']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Dataset")
            fig1 = plot_class_distribution(labels)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Explained Variance")
            fig2 = plot_explained_variance(pca)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Scatter Plot Ruang PCA")
        fig3 = plot_pca_scatter(X_pca, labels)
        st.plotly_chart(fig3, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Eigenfaces")
            if 'X_raw' in st.session_state:
                fig4 = plot_eigenfaces(pca)
                st.pyplot(fig4)
        
        with col4:
            st.subheader("Mean Face")
            if 'X_raw' in st.session_state:
                fig5 = plot_mean_face(st.session_state['X_raw'])
                st.pyplot(fig5)

# ===================== TAB 3: BANDINGKAN 2 WAJAH =====================
with tabs[2]:
    st.header("🔍 Bandingkan Dua Wajah")
    
    col1, col2 = st.columns(2)
    with col1:
        upload1 = st.file_uploader("Upload Wajah 1", type=['jpg','jpeg','png'],
                                    key="face1")
        if upload1:
            img1 = Image.open(upload1)
            st.image(img1, caption="Wajah 1", use_column_width=True)
    
    with col2:
        upload2 = st.file_uploader("Upload Wajah 2", type=['jpg','jpeg','png'],
                                    key="face2")
        if upload2:
            img2 = Image.open(upload2)
            st.image(img2, caption="Wajah 2", use_column_width=True)
    
    if upload1 and upload2 and 'pca' in st.session_state:
        if st.button("🔍 Bandingkan", type="primary"):
            pca = st.session_state['pca']
            
            # Convert PIL ke numpy
            arr1 = np.array(img1.convert('RGB'))
            arr2 = np.array(img2.convert('RGB'))
            arr1_bgr = cv2.cvtColor(arr1, cv2.COLOR_RGB2BGR)
            arr2_bgr = cv2.cvtColor(arr2, cv2.COLOR_RGB2BGR)
            
            vec1, detected1 = detect_and_crop_face(image_array=arr1_bgr)
            vec2, detected2 = detect_and_crop_face(image_array=arr2_bgr)
            
            similarity = compare_two_faces(vec1, vec2, pca)
            
            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Similarity Score", f"{similarity:.4f}")
            with col_r2:
                st.metric("Threshold", f"{threshold:.2f}")
            with col_r3:
                if similarity >= threshold:
                    st.success("✅ MIRIP")
                else:
                    st.error("❌ TIDAK MIRIP")
            
            # Progress bar similarity
            st.progress(float(similarity), text=f"Similarity: {similarity:.2%}")
            
            # Info deteksi wajah
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                icon = "✅" if detected1 else "⚠️"
                st.info(f"{icon} Wajah 1: "
                        f"{'terdeteksi' if detected1 else 'fallback ke full image'}")
            with col_d2:
                icon = "✅" if detected2 else "⚠️"
                st.info(f"{icon} Wajah 2: "
                        f"{'terdeteksi' if detected2 else 'fallback ke full image'}")

# ===================== TAB 4: IDENTIFIKASI =====================
with tabs[3]:
    st.header("🎯 Identifikasi Wajah dari Database")
    
    upload_id = st.file_uploader("Upload foto wajah untuk diidentifikasi",
                                  type=['jpg','jpeg','png'], key="identify")
    
    if upload_id and 'pca' in st.session_state:
        img_id = Image.open(upload_id)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_id, caption="Foto Input", use_column_width=True)
        
        with col2:
            if st.button("🎯 Identifikasi", type="primary"):
                pca = st.session_state['pca']
                X_pca = st.session_state['X_pca']
                labels = st.session_state['labels']
                
                arr = np.array(img_id.convert('RGB'))
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                vec, detected = detect_and_crop_face(image_array=arr_bgr)
                
                name, score, all_sims = recognize_face(
                    vec, pca, X_pca, labels, threshold
                )
                
                st.subheader("Hasil Identifikasi")
                if name != "Tidak dikenal":
                    st.success(f"✅ **{name}** (similarity: {score:.4f})")
                else:
                    st.error(f"❌ Tidak dikenal (best score: {score:.4f})")
                
                # Top 5 paling mirip
                unique_labels = list(set(labels))
                top_idx = np.argsort(all_sims)[::-1][:10]
                
                import pandas as pd
                df_top = pd.DataFrame({
                    'Person': [labels[i] for i in top_idx],
                    'Similarity': [all_sims[i] for i in top_idx]
                }).drop_duplicates('Person').head(5)
                
                st.subheader("Top 5 Paling Mirip")
                st.dataframe(df_top, use_container_width=True)

# ===================== TAB 5: MASA KECIL vs DEWASA =====================
with tabs[4]:
    st.header("👶 Foto Masa Kecil vs Foto Dewasa")
    st.markdown("""
    Upload foto kamu waktu kecil dan foto kamu sekarang.  
    Sistem akan menghitung seberapa mirip wajahmu antara dua waktu berbeda.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Foto Masa Kecil")
        child_photo = st.file_uploader("Upload foto masa kecil",
                                        type=['jpg','jpeg','png'], key="child")
        if child_photo:
            img_child = Image.open(child_photo)
            st.image(img_child, use_column_width=True)
    
    with col2:
        st.subheader("🧑 Foto Sekarang (Dewasa)")
        adult_photo = st.file_uploader("Upload foto sekarang",
                                        type=['jpg','jpeg','png'], key="adult")
        if adult_photo:
            img_adult = Image.open(adult_photo)
            st.image(img_adult, use_column_width=True)
    
    if child_photo and adult_photo and 'pca' in st.session_state:
        if st.button("🔬 Analisis Kemiripan", type="primary"):
            pca = st.session_state['pca']
            
            arr_c = cv2.cvtColor(np.array(img_child.convert('RGB')),
                                  cv2.COLOR_RGB2BGR)
            arr_a = cv2.cvtColor(np.array(img_adult.convert('RGB')),
                                  cv2.COLOR_RGB2BGR)
            
            vec_c, det_c = detect_and_crop_face(image_array=arr_c)
            vec_a, det_a = detect_and_crop_face(image_array=arr_a)
            
            similarity = compare_two_faces(vec_c, vec_a, pca)
            
            st.divider()
            st.subheader("📊 Hasil Analisis")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Similarity Score", f"{similarity:.4f}")
            with cols[1]:
                st.metric("Persentase Kemiripan", f"{similarity*100:.1f}%")
            with cols[2]:
                if similarity >= threshold:
                    st.success("✅ KEMUNGKINAN ORANG YANG SAMA")
                else:
                    st.warning("⚠️ KURANG MIRIP (wajar karena perbedaan usia)")
            
            st.progress(float(max(0, similarity)))
            
            st.info("""
            💡 **Catatan:** PCA/SVD sensitif terhadap perubahan fisik akibat pertumbuhan usia.  
            Nilai similarity yang lebih rendah dari foto orang dewasa adalah hal yang wajar.  
            Metode deep learning (FaceNet/ArcFace) lebih baik untuk kasus ini.
=======
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import joblib

from preprocessing import detect_and_crop_face, load_and_preprocess
from pca_model import (load_dataset, train_pca, compare_two_faces,
                       recognize_face, save_model, load_model)
from eda import (plot_class_distribution, plot_explained_variance,
                 plot_eigenfaces, plot_pca_scatter, plot_mean_face)

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Face Similarity PCA",
    page_icon="👤",
    layout="wide"
)

MODEL_PATH = "models/pca_model.pkl"

# ===================== SIDEBAR =====================
st.sidebar.title("⚙️ Konfigurasi")
n_components = st.sidebar.slider("Jumlah Komponen PCA", 10, 150, 50)
threshold = st.sidebar.slider("Threshold Similarity", 0.5, 1.0, 0.80, 0.01)
dataset_path = st.sidebar.text_input("Path Dataset Train", "dataset/Train")

# ===================== MAIN =====================
st.title("👤 Face Similarity Detection")
st.markdown("**Metode:** PCA/SVD (Eigenfaces) | **Framework:** Streamlit")

tabs = st.tabs(["🏋️ Training", "📊 EDA", 
                "🔍 Bandingkan 2 Wajah", "🎯 Identifikasi Wajah",
                "👶 Masa Kecil vs Dewasa"])

# ===================== TAB 1: TRAINING =====================
with tabs[0]:
    st.header("Training Model PCA")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Dataset path: `{dataset_path}`")
        if st.button("🚀 Mulai Training", type="primary"):
            with st.spinner("Memuat dataset..."):
                if not os.path.exists(dataset_path):
                    st.error("Folder dataset tidak ditemukan!")
                    st.stop()
                
                X, labels, paths = load_dataset(dataset_path)
                st.success(f"✅ Dataset dimuat: {len(X)} gambar, "
                           f"{len(set(labels))} orang")
            
            with st.spinner("Melatih PCA..."):
                pca, X_pca = train_pca(X, n_components=n_components)
                save_model(pca, X_pca, labels)
                
                st.session_state['pca'] = pca
                st.session_state['X_pca'] = X_pca
                st.session_state['labels'] = labels
                st.session_state['X_raw'] = X
                
                total_var = np.sum(pca.explained_variance_ratio_) * 100
                st.success(f"✅ PCA selesai! Total variance: {total_var:.1f}%")
    
    with col2:
        st.subheader("Info Model")
        if 'pca' in st.session_state:
            pca = st.session_state['pca']
            st.metric("Jumlah Komponen", n_components)
            st.metric("Total Gambar", len(st.session_state['labels']))
            st.metric("Total Orang", len(set(st.session_state['labels'])))
            st.metric("Explained Variance",
                      f"{np.sum(pca.explained_variance_ratio_)*100:.1f}%")
        else:
            # Coba load model tersimpan
            if os.path.exists(MODEL_PATH):
                pca, X_pca, labels = load_model(MODEL_PATH)
                st.session_state['pca'] = pca
                st.session_state['X_pca'] = X_pca
                st.session_state['labels'] = labels
                st.info("Model sebelumnya berhasil dimuat!")

# ===================== TAB 2: EDA =====================
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    
    if 'pca' not in st.session_state:
        st.warning("⚠️ Latih model terlebih dahulu di tab Training!")
    else:
        pca = st.session_state['pca']
        X_pca = st.session_state['X_pca']
        labels = st.session_state['labels']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Dataset")
            fig1 = plot_class_distribution(labels)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Explained Variance")
            fig2 = plot_explained_variance(pca)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Scatter Plot Ruang PCA")
        fig3 = plot_pca_scatter(X_pca, labels)
        st.plotly_chart(fig3, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Eigenfaces")
            if 'X_raw' in st.session_state:
                fig4 = plot_eigenfaces(pca)
                st.pyplot(fig4)
        
        with col4:
            st.subheader("Mean Face")
            if 'X_raw' in st.session_state:
                fig5 = plot_mean_face(st.session_state['X_raw'])
                st.pyplot(fig5)

# ===================== TAB 3: BANDINGKAN 2 WAJAH =====================
with tabs[2]:
    st.header("🔍 Bandingkan Dua Wajah")
    
    col1, col2 = st.columns(2)
    with col1:
        upload1 = st.file_uploader("Upload Wajah 1", type=['jpg','jpeg','png'],
                                    key="face1")
        if upload1:
            img1 = Image.open(upload1)
            st.image(img1, caption="Wajah 1", use_column_width=True)
    
    with col2:
        upload2 = st.file_uploader("Upload Wajah 2", type=['jpg','jpeg','png'],
                                    key="face2")
        if upload2:
            img2 = Image.open(upload2)
            st.image(img2, caption="Wajah 2", use_column_width=True)
    
    if upload1 and upload2 and 'pca' in st.session_state:
        if st.button("🔍 Bandingkan", type="primary"):
            pca = st.session_state['pca']
            
            # Convert PIL ke numpy
            arr1 = np.array(img1.convert('RGB'))
            arr2 = np.array(img2.convert('RGB'))
            arr1_bgr = cv2.cvtColor(arr1, cv2.COLOR_RGB2BGR)
            arr2_bgr = cv2.cvtColor(arr2, cv2.COLOR_RGB2BGR)
            
            vec1, detected1 = detect_and_crop_face(image_array=arr1_bgr)
            vec2, detected2 = detect_and_crop_face(image_array=arr2_bgr)
            
            similarity = compare_two_faces(vec1, vec2, pca)
            
            st.divider()
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Similarity Score", f"{similarity:.4f}")
            with col_r2:
                st.metric("Threshold", f"{threshold:.2f}")
            with col_r3:
                if similarity >= threshold:
                    st.success("✅ MIRIP")
                else:
                    st.error("❌ TIDAK MIRIP")
            
            # Progress bar similarity
            st.progress(float(similarity), text=f"Similarity: {similarity:.2%}")
            
            # Info deteksi wajah
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                icon = "✅" if detected1 else "⚠️"
                st.info(f"{icon} Wajah 1: "
                        f"{'terdeteksi' if detected1 else 'fallback ke full image'}")
            with col_d2:
                icon = "✅" if detected2 else "⚠️"
                st.info(f"{icon} Wajah 2: "
                        f"{'terdeteksi' if detected2 else 'fallback ke full image'}")

# ===================== TAB 4: IDENTIFIKASI =====================
with tabs[3]:
    st.header("🎯 Identifikasi Wajah dari Database")
    
    upload_id = st.file_uploader("Upload foto wajah untuk diidentifikasi",
                                  type=['jpg','jpeg','png'], key="identify")
    
    if upload_id and 'pca' in st.session_state:
        img_id = Image.open(upload_id)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_id, caption="Foto Input", use_column_width=True)
        
        with col2:
            if st.button("🎯 Identifikasi", type="primary"):
                pca = st.session_state['pca']
                X_pca = st.session_state['X_pca']
                labels = st.session_state['labels']
                
                arr = np.array(img_id.convert('RGB'))
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                vec, detected = detect_and_crop_face(image_array=arr_bgr)
                
                name, score, all_sims = recognize_face(
                    vec, pca, X_pca, labels, threshold
                )
                
                st.subheader("Hasil Identifikasi")
                if name != "Tidak dikenal":
                    st.success(f"✅ **{name}** (similarity: {score:.4f})")
                else:
                    st.error(f"❌ Tidak dikenal (best score: {score:.4f})")
                
                # Top 5 paling mirip
                unique_labels = list(set(labels))
                top_idx = np.argsort(all_sims)[::-1][:10]
                
                import pandas as pd
                df_top = pd.DataFrame({
                    'Person': [labels[i] for i in top_idx],
                    'Similarity': [all_sims[i] for i in top_idx]
                }).drop_duplicates('Person').head(5)
                
                st.subheader("Top 5 Paling Mirip")
                st.dataframe(df_top, use_container_width=True)

# ===================== TAB 5: MASA KECIL vs DEWASA =====================
with tabs[4]:
    st.header("👶 Foto Masa Kecil vs Foto Dewasa")
    st.markdown("""
    Upload foto kamu waktu kecil dan foto kamu sekarang.  
    Sistem akan menghitung seberapa mirip wajahmu antara dua waktu berbeda.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Foto Masa Kecil")
        child_photo = st.file_uploader("Upload foto masa kecil",
                                        type=['jpg','jpeg','png'], key="child")
        if child_photo:
            img_child = Image.open(child_photo)
            st.image(img_child, use_column_width=True)
    
    with col2:
        st.subheader("🧑 Foto Sekarang (Dewasa)")
        adult_photo = st.file_uploader("Upload foto sekarang",
                                        type=['jpg','jpeg','png'], key="adult")
        if adult_photo:
            img_adult = Image.open(adult_photo)
            st.image(img_adult, use_column_width=True)
    
    if child_photo and adult_photo and 'pca' in st.session_state:
        if st.button("🔬 Analisis Kemiripan", type="primary"):
            pca = st.session_state['pca']
            
            arr_c = cv2.cvtColor(np.array(img_child.convert('RGB')),
                                  cv2.COLOR_RGB2BGR)
            arr_a = cv2.cvtColor(np.array(img_adult.convert('RGB')),
                                  cv2.COLOR_RGB2BGR)
            
            vec_c, det_c = detect_and_crop_face(image_array=arr_c)
            vec_a, det_a = detect_and_crop_face(image_array=arr_a)
            
            similarity = compare_two_faces(vec_c, vec_a, pca)
            
            st.divider()
            st.subheader("📊 Hasil Analisis")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Similarity Score", f"{similarity:.4f}")
            with cols[1]:
                st.metric("Persentase Kemiripan", f"{similarity*100:.1f}%")
            with cols[2]:
                if similarity >= threshold:
                    st.success("✅ KEMUNGKINAN ORANG YANG SAMA")
                else:
                    st.warning("⚠️ KURANG MIRIP (wajar karena perbedaan usia)")
            
            st.progress(float(max(0, similarity)))
            
            st.info(""" 
            Nilai similarity yang lebih rendah dari foto orang dewasa adalah hal yang wajar.  
            Metode deep learning (FaceNet/ArcFace) lebih baik untuk kasus ini.
            """)
