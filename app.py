import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Mask Detector",
    page_icon="😷",
    layout="wide"
)

# Judul aplikasi
st.title("🎭 Real-Time Mask Detector")
st.markdown("**Deteksi penggunaan masker secara real-time menggunakan webcam**")

# Sidebar untuk informasi
st.sidebar.title("ℹ️ Informasi")
st.sidebar.markdown("""
- **Hijau**: Menggunakan masker ✅
- **Merah**: Tidak menggunakan masker ❌
- Tekan 'START' untuk memulai deteksi
- Tekan 'STOP' untuk menghentikan
""")

# Load model dan cascade classifier
def load_model():
    try:
        model_path = 'mask_detector_svm.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.sidebar.success("✅ Model berhasil dimuat!")
            st.sidebar.info(f"📁 File: {model_path}")
            return model
        else:
            st.sidebar.error("❌ File model tidak ditemukan!")
            st.sidebar.info("Silakan jalankan train_mask_detector.py terlebih dahulu")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_face_cascade():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

# Class untuk video transformer
class MaskDetectorTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.face_cascade = load_face_cascade()
        self.IMG_SIZE = 100
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.model is None or self.face_cascade is None:
            # Jika model tidak ada, tampilkan frame asli dengan pesan
            cv2.putText(img, "Model tidak tersedia", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
        
        # Deteksi wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Proses setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Extract wajah
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (self.IMG_SIZE, self.IMG_SIZE))
            face_flat = face_resized.flatten().reshape(1, -1)
            
            # Prediksi
            try:
                prediction = self.model.predict(face_flat)[0]
                confidence = self.model.decision_function(face_flat)[0]
                
                # Label dan warna
                if prediction == 0:  # Dengan masker
                    label = "Mask"
                    color = (0, 255, 0)  # Hijau
                    emoji = "😷"
                else:  # Tanpa masker
                    label = "No Mask"
                    color = (0, 0, 255)  # Merah
                    emoji = "😐"
                
                # Gambar rectangle dan text
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                
                # Label dengan emoji
                label_text = f"{emoji} {label}"
                cv2.putText(img, label_text, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Confidence score
                conf_text = f"Conf: {abs(confidence):.2f}"
                cv2.putText(img, conf_text, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            except Exception as e:
                # Jika error dalam prediksi
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                cv2.putText(img, "Error", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Tambahkan informasi di pojok kiri atas
        cv2.putText(img, f"Wajah terdeteksi: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img

# Konfigurasi RTC untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Fungsi untuk cek dataset
def check_dataset():
    """Cek apakah dataset sudah tersedia"""
    dataset_path = "dataset"
    with_mask_path = f"{dataset_path}/with_mask"
    without_mask_path = f"{dataset_path}/without_mask"

    if not os.path.exists(dataset_path):
        return False, 0, 0

    if not os.path.exists(with_mask_path) or not os.path.exists(without_mask_path):
        return False, 0, 0

    import glob
    with_mask_files = glob.glob(f'{with_mask_path}/*.jpg')
    without_mask_files = glob.glob(f'{without_mask_path}/*.jpg')

    return len(with_mask_files) > 0 and len(without_mask_files) > 0, len(with_mask_files), len(without_mask_files)

# Main app
def main():
    # Cek apakah model tersedia
    model = load_model()

    # Cek dataset
    has_dataset, with_mask_count, without_mask_count = check_dataset()

    # Tab untuk berbagai mode
    tab1, tab2, tab3 = st.tabs(["🎥 Real-Time Detection", "📁 Upload Image", "ℹ️ Setup Guide"])

    with tab1:
        # Tombol untuk force reload model
        col_refresh1, col_refresh2 = st.columns([1, 4])
        with col_refresh1:
            if st.button("🔄 Reload Model"):
                st.cache_data.clear()
                st.rerun()

        if model is None:
            st.error("🚨 **Model tidak ditemukan!**")

            # Debug info
            st.info("🔍 **Debug Info:**")
            st.write(f"- File exists: {os.path.exists('mask_detector_svm.pkl')}")
            if os.path.exists('mask_detector_svm.pkl'):
                st.write(f"- File size: {os.path.getsize('mask_detector_svm.pkl'):,} bytes")

            if not has_dataset:
                st.warning("📦 **Dataset belum tersedia**")
                st.info("**Langkah-langkah untuk menjalankan aplikasi:**")

                # Tombol download dataset
                st.markdown("### 1. Download Dataset")
                st.markdown("Klik link berikut untuk download dataset:")
                st.markdown("🔗 [Download Dataset](https://drive.google.com/file/d/191ugrhUoIAuN3Q6To4goadPMIgQMzt0U/view?usp=sharing)")

                st.markdown("### 2. Ekstrak Dataset")
                st.code("""
Ekstrak file zip ke folder:
dataset/
├── with_mask/     ← Gambar orang pakai masker
└── without_mask/  ← Gambar orang tanpa masker
                """)

                st.markdown("### 3. Training Model")
                st.code("python train_mask_detector.py")

                st.markdown("### 4. Refresh Halaman")
                st.info("Setelah training selesai, refresh halaman ini")

            else:
                st.success(f"✅ Dataset tersedia: {with_mask_count} with_mask, {without_mask_count} without_mask")
                st.info("**Jalankan training untuk membuat model:**")
                st.code("python train_mask_detector.py")

                if st.button("🔄 Refresh untuk cek model"):
                    st.rerun()
        else:
            st.subheader("🎭 Real-Time Mask Detection")
            # Kolom untuk layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📹 Live Camera Feed")

                # WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key="mask-detector-main",
                    video_transformer_factory=MaskDetectorTransformer,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={
                        "video": True,
                        "audio": False
                    },
                    async_processing=True,
                )

            with col2:
                st.subheader("📊 Status")

                if webrtc_ctx.video_transformer:
                    st.success("🟢 Kamera aktif")
                    st.info("🔍 Deteksi berjalan...")
                else:
                    st.warning("🟡 Kamera tidak aktif")
                    st.info("👆 Klik 'START' untuk memulai")

    with tab2:
        st.subheader("📁 Upload Gambar untuk Deteksi")

        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar yang berisi wajah untuk deteksi mask"
        )

        if uploaded_file is not None:
            # Baca gambar
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Asli")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col2:
                st.subheader("🔍 Hasil Deteksi")

                if model is not None:
                    # Proses deteksi
                    face_cascade = load_face_cascade()
                    if face_cascade is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                        result_img = img.copy()

                        for (x, y, w, h) in faces:
                            face = img[y:y+h, x:x+w]
                            face_resized = cv2.resize(face, (100, 100))
                            face_flat = face_resized.flatten().reshape(1, -1)

                            prediction = model.predict(face_flat)[0]
                            confidence = model.decision_function(face_flat)[0]

                            if prediction == 0:
                                label = "Mask"
                                color = (0, 255, 0)
                                emoji = "😷"
                            else:
                                label = "No Mask"
                                color = (0, 0, 255)
                                emoji = "😐"

                            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
                            cv2.putText(result_img, f"{emoji} {label}", (x, y-15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                        st.success(f"✅ Terdeteksi {len(faces)} wajah")
                    else:
                        st.error("❌ Face cascade tidak tersedia")
                else:
                    st.error("❌ Model tidak tersedia")
                    st.info("Silakan download dataset dan jalankan training terlebih dahulu")

    with tab3:
        st.subheader("📋 Setup Guide")
        st.markdown("""
        ### Langkah-langkah untuk mengaktifkan deteksi mask:

        1. **Download Dataset**
           - Klik link di README.md untuk download dataset
           - Ekstrak ke folder `dataset/` dengan struktur:
           ```
           dataset/
           ├── with_mask/
           └── without_mask/
           ```

        2. **Training Model**
           ```bash
           python train_mask_detector.py
           ```
           - Proses ini akan membuat file `mask_detector_svm.pkl`

        3. **Refresh Aplikasi**
           - Setelah training selesai, refresh halaman ini
           - Model akan otomatis terdeteksi dan dimuat

        ### Struktur Folder yang Benar:
        """)

        st.code("""
        mask-detector/
        ├── dataset/
        │   ├── with_mask/        ← Gambar orang pakai masker
        │   └── without_mask/     ← Gambar orang tanpa masker
        ├── train_mask_detector.py
        ├── detect_mask_webcam.py
        ├── app.py
        ├── requirements.txt
        └── mask_detector_svm.pkl  ← File ini dibuat setelah training
        """)

        st.markdown("""
        ### Troubleshooting:
        - **Kamera tidak muncul**: Pastikan browser mengizinkan akses kamera
        - **Model tidak terdeteksi**: Pastikan file `mask_detector_svm.pkl` ada
        - **Error saat training**: Pastikan dataset sudah di-download dan diekstrak dengan benar
        """)

        # Informasi tambahan
        st.subheader("🎯 Cara Penggunaan")
        st.markdown("""
        1. **Klik tombol START** untuk mengaktifkan kamera
        2. **Posisikan wajah** di depan kamera
        3. **Lihat hasil deteksi** secara real-time:
           - 🟢 **Hijau**: Menggunakan masker
           - 🔴 **Merah**: Tidak menggunakan masker
        4. **Klik STOP** untuk menghentikan
        """)

        # Statistik model (jika tersedia)
        st.subheader("🤖 Info Model")
        st.info("Model: Support Vector Machine (SVM)")
        st.info("Input: Gambar 100x100 pixels")
        st.info("Framework: scikit-learn + OpenCV")

if __name__ == "__main__":
    main()
