# 🎭 Real-Time Mask Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

**Pengolahan Citra - Project**

Aplikasi deteksi masker real-time menggunakan webcam dengan teknologi machine learning (SVM) dan computer vision (OpenCV).

## 🌐 Live Demo

**[🚀 Try the Live App](https://your-app-url.streamlit.app)**

> **Note**: Ganti `your-app-url` dengan URL aplikasi Streamlit Anda setelah deployment

## 🌟 Fitur

- ✅ **Real-time detection** menggunakan webcam
- ✅ **Upload gambar** untuk deteksi batch
- ✅ **Web interface** yang user-friendly
- ✅ **Deploy ke cloud** dengan Streamlit Cloud
- ✅ **Demo mode** tanpa model (hanya deteksi wajah)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd mask-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download dataset dari link berikut:
https://drive.google.com/file/d/191ugrhUoIAuN3Q6To4goadPMIgQMzt0U/view?usp=sharing

### 4. Setup Dataset

Ekstrak dataset dengan struktur folder:

```
dataset/
├── with_mask/     ← Gambar orang pakai masker
└── without_mask/  ← Gambar orang tanpa masker
```

### 5. Training Model

```bash
python train_mask_detector.py
```

File `mask_detector_svm.pkl` akan dibuat setelah training selesai.

### 6. Jalankan Aplikasi

#### Desktop Version (Original)

```bash
python detect_mask_webcam.py
```

#### Web Version (Streamlit)

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## 📁 Struktur Project

```
mask-detector/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── .streamlit/
│   └── config.toml
├── train_mask_detector.py    ← Training script
├── detect_mask_webcam.py     ← Desktop version
├── app.py                    ← Web version (Streamlit)
├── requirements.txt          ← Python dependencies
├── packages.txt              ← System dependencies
├── mask_detector_svm.pkl     ← Trained model (dibuat setelah training)
└── README.md
```

## 🌐 Deploy ke Cloud

### Streamlit Cloud (Recommended)

1. Push code ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account
4. Deploy dari repository
5. **Important**: Anda perlu upload model `mask_detector_svm.pkl` atau train ulang di cloud
6. Aplikasi akan otomatis tersedia di URL public

### Platform Lain

- **Hugging Face Spaces**: Upload sebagai Streamlit Space
- **Railway**: Deploy dengan auto-detection
- **Render**: Deploy sebagai web service

### ⚠️ Important Notes untuk Deployment

1. **Model File**: File `mask_detector_svm.pkl` (~472MB) terlalu besar untuk GitHub
2. **Dataset**: Folder `dataset/` (~7GB) tidak di-upload ke repository
3. **Solusi**:
   - Upload model ke cloud storage (Google Drive, Dropbox, dll)
   - Atau train ulang model di cloud environment
   - Atau gunakan model yang lebih kecil

## 🎯 Cara Penggunaan Web App

### Tab 1: Real-Time Detection

1. Klik tombol **START** untuk mengaktifkan kamera
2. Posisikan wajah di depan kamera
3. Lihat hasil deteksi real-time:
   - 🟢 **Hijau**: Menggunakan masker
   - 🔴 **Merah**: Tidak menggunakan masker
4. Klik **STOP** untuk menghentikan

### Tab 2: Upload Image

1. Upload gambar (JPG, PNG)
2. Lihat hasil deteksi pada gambar
3. Download hasil jika diperlukan

### Tab 3: Setup Guide

- Panduan lengkap setup dan troubleshooting
- Informasi struktur folder
- Tips penggunaan

## 🤖 Technical Details

### Model

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Linear
- **Input**: 100x100 pixel images (flattened)
- **Classes**: 0 (with mask), 1 (without mask)

### Dependencies

- **Streamlit**: Web framework
- **OpenCV**: Computer vision
- **scikit-learn**: Machine learning
- **streamlit-webrtc**: Real-time video processing

## 🔧 Troubleshooting

### Kamera tidak muncul

- Pastikan browser mengizinkan akses kamera
- Coba refresh halaman
- Periksa permission di browser settings

### Model tidak terdeteksi

- Pastikan file `mask_detector_svm.pkl` ada
- Jalankan training ulang jika diperlukan
- Periksa path file model

### Error saat training

- Pastikan dataset sudah di-download
- Periksa struktur folder dataset
- Pastikan ada gambar di folder `with_mask` dan `without_mask`

## 📊 Performance

- **Accuracy**: ~95% (tergantung kualitas dataset)
- **Speed**: Real-time (30+ FPS)
- **Model Size**: ~50MB

## 🎓 Credits

**Tugas Pengolahan Citra**

- Machine Learning dengan SVM
- Computer Vision dengan OpenCV
- Web Development dengan Streamlit
