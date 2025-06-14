#!/usr/bin/env python3
"""
Model downloader untuk Streamlit Cloud
Download model dari cloud storage secara otomatis
"""

import os
import requests
import streamlit as st
import joblib
from pathlib import Path

def download_model_from_url(url, filename="mask_detector_svm.pkl"):
    """Download model dari URL cloud storage"""
    try:
        if os.path.exists(filename):
            st.info(f"✅ Model {filename} sudah ada")
            return True
            
        st.info(f"📥 Downloading model dari cloud storage...")
        
        # Download dengan progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = downloaded / total_size
                        st.progress(progress)
        
        st.success(f"✅ Model berhasil didownload: {filename}")
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return False

@st.cache_data
def load_model_from_cloud(model_url):
    """Load model dengan caching"""
    try:
        # Download jika belum ada
        if download_model_from_url(model_url):
            # Load model
            model = joblib.load("mask_detector_svm.pkl")
            return model
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Contoh penggunaan:
# MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
# model = load_model_from_cloud(MODEL_URL)
