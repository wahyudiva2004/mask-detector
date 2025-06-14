#!/usr/bin/env python3
"""
Script untuk membantu deployment aplikasi Mask Detector
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_requirements():
    """Cek apakah requirements sudah terinstall"""
    try:
        import streamlit
        import cv2
        import sklearn
        import joblib
        print("✅ Semua dependencies sudah terinstall")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def install_requirements():
    """Install requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements berhasil diinstall")
        return True
    except subprocess.CalledProcessError:
        print("❌ Gagal install requirements")
        return False

def check_model():
    """Cek apakah model sudah ada"""
    model_path = Path("mask_detector_svm.pkl")
    if model_path.exists():
        print("✅ Model ditemukan")
        return True
    else:
        print("⚠️ Model tidak ditemukan")
        return False

def check_dataset():
    """Cek apakah dataset sudah ada"""
    dataset_path = Path("dataset")
    with_mask = dataset_path / "with_mask"
    without_mask = dataset_path / "without_mask"
    
    if dataset_path.exists() and with_mask.exists() and without_mask.exists():
        with_mask_count = len(list(with_mask.glob("*.jpg")))
        without_mask_count = len(list(without_mask.glob("*.jpg")))
        print(f"✅ Dataset ditemukan: {with_mask_count} with_mask, {without_mask_count} without_mask")
        return True
    else:
        print("⚠️ Dataset tidak ditemukan")
        return False

def train_model():
    """Jalankan training model"""
    print("🤖 Training model...")
    try:
        subprocess.check_call([sys.executable, "train_mask_detector.py"])
        print("✅ Training selesai")
        return True
    except subprocess.CalledProcessError:
        print("❌ Training gagal")
        return False

def run_streamlit():
    """Jalankan aplikasi Streamlit"""
    print("🚀 Menjalankan aplikasi Streamlit...")
    try:
        # Buka browser otomatis
        webbrowser.open("http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplikasi dihentikan")

def main():
    print("🎭 Mask Detector Deployment Script")
    print("=" * 40)
    
    # Cek requirements
    if not check_requirements():
        print("\n📦 Installing requirements...")
        if not install_requirements():
            print("❌ Gagal install requirements. Silakan install manual:")
            print("pip install -r requirements.txt")
            return
    
    # Cek dataset
    has_dataset = check_dataset()
    
    # Cek model
    has_model = check_model()
    
    # Jika tidak ada model tapi ada dataset, tawarkan training
    if not has_model and has_dataset:
        response = input("\n🤖 Model tidak ada. Jalankan training? (y/n): ")
        if response.lower() == 'y':
            if train_model():
                has_model = True
    
    # Info status
    print("\n📊 Status:")
    print(f"   Dependencies: {'✅' if check_requirements() else '❌'}")
    print(f"   Dataset: {'✅' if has_dataset else '⚠️'}")
    print(f"   Model: {'✅' if has_model else '⚠️'}")
    
    if not has_model:
        print("\n⚠️ Aplikasi akan berjalan dalam mode demo (tanpa klasifikasi mask)")
    
    # Jalankan aplikasi
    print("\n🚀 Menjalankan aplikasi...")
    run_streamlit()

if __name__ == "__main__":
    main()
