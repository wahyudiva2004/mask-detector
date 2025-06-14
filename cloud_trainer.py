#!/usr/bin/env python3
"""
Cloud trainer untuk Streamlit Cloud
Train model secara real-time dari uploaded dataset
"""

import streamlit as st
import zipfile
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import tempfile

def extract_uploaded_dataset(uploaded_zip):
    """Extract dataset dari uploaded ZIP file"""
    try:
        # Buat temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        return temp_dir
    except Exception as e:
        st.error(f"❌ Error extracting dataset: {str(e)}")
        return None

def load_images_from_folder(folder_path, label):
    """Load gambar dari folder"""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        return images, labels
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (100, 100))
                    img_flat = img_resized.flatten()
                    images.append(img_flat)
                    labels.append(label)
            except Exception as e:
                continue
    
    return images, labels

@st.cache_data
def train_model_from_dataset(temp_dir):
    """Train model dari dataset yang di-upload"""
    try:
        st.info("🤖 Memulai training model...")
        
        # Load images
        with_mask_path = os.path.join(temp_dir, "with_mask")
        without_mask_path = os.path.join(temp_dir, "without_mask")
        
        # Load with_mask images (label 0)
        with_mask_images, with_mask_labels = load_images_from_folder(with_mask_path, 0)
        st.info(f"📁 Loaded {len(with_mask_images)} with_mask images")
        
        # Load without_mask images (label 1)  
        without_mask_images, without_mask_labels = load_images_from_folder(without_mask_path, 1)
        st.info(f"📁 Loaded {len(without_mask_images)} without_mask images")
        
        if len(with_mask_images) == 0 or len(without_mask_images) == 0:
            st.error("❌ Dataset tidak lengkap! Pastikan ada folder with_mask dan without_mask")
            return None
        
        # Combine data
        X = np.array(with_mask_images + without_mask_images)
        y = np.array(with_mask_labels + without_mask_labels)
        
        st.info(f"📊 Total dataset: {len(X)} gambar")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        st.info("🔄 Training SVM model...")
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"✅ Model berhasil di-train!")
        st.success(f"📊 Akurasi: {accuracy*100:.2f}%")
        
        # Save model
        joblib.dump(model, 'mask_detector_svm.pkl')
        st.success("💾 Model berhasil disimpan!")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None

def show_training_interface():
    """Interface untuk upload dataset dan training"""
    st.subheader("🎯 Train Model dari Dataset")
    
    st.markdown("""
    **Langkah-langkah:**
    1. Siapkan dataset dalam format ZIP
    2. Struktur folder: `dataset/with_mask/` dan `dataset/without_mask/`
    3. Upload ZIP file di bawah ini
    4. Klik tombol "Train Model"
    """)
    
    uploaded_zip = st.file_uploader(
        "Upload Dataset (ZIP format)",
        type=['zip'],
        help="Upload file ZIP yang berisi folder with_mask dan without_mask"
    )
    
    if uploaded_zip is not None:
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Extract dataset
                temp_dir = extract_uploaded_dataset(uploaded_zip)
                
                if temp_dir:
                    # Train model
                    model = train_model_from_dataset(temp_dir)
                    
                    if model:
                        st.balloons()
                        st.success("🎉 Model berhasil di-train dan siap digunakan!")
                        st.info("🔄 Refresh halaman untuk menggunakan model baru")
                    
                    # Cleanup
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
