#!/usr/bin/env python3
"""
Model storage menggunakan Streamlit Secrets
Untuk model kecil yang bisa di-encode ke Base64
"""

import streamlit as st
import base64
import joblib
import io
import pickle

def encode_model_to_base64(model_path):
    """Encode model file ke Base64 string"""
    try:
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        
        encoded = base64.b64encode(model_bytes).decode('utf-8')
        
        print(f"Model size: {len(model_bytes)} bytes")
        print(f"Encoded size: {len(encoded)} characters")
        print(f"Add this to your .streamlit/secrets.toml:")
        print(f'model_base64 = "{encoded}"')
        
        return encoded
    except Exception as e:
        print(f"Error encoding model: {e}")
        return None

@st.cache_data
def load_model_from_secrets():
    """Load model dari Streamlit Secrets"""
    try:
        # Get encoded model from secrets
        if "model_base64" not in st.secrets:
            st.error("❌ Model tidak ditemukan di secrets")
            return None
        
        encoded_model = st.secrets["model_base64"]
        
        # Decode Base64
        model_bytes = base64.b64decode(encoded_model)
        
        # Load model dari bytes
        model = joblib.load(io.BytesIO(model_bytes))
        
        st.success("✅ Model berhasil dimuat dari secrets!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model from secrets: {str(e)}")
        return None

# Contoh penggunaan untuk encode model:
# python secrets_model.py
if __name__ == "__main__":
    # Encode model yang sudah ada
    model_path = "mask_detector_svm.pkl"
    if os.path.exists(model_path):
        encode_model_to_base64(model_path)
    else:
        print("Model file tidak ditemukan!")
