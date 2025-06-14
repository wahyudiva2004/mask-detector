import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os
import time
import requests
import gdown

# Konfigurasi halaman
st.set_page_config(
    page_title="🎭 Mask Detector",
    page_icon="😷",
    layout="centered"
)

# Judul aplikasi
st.title("🎭 Real-Time Mask Detector")
st.markdown("**Deteksi penggunaan masker secara real-time menggunakan webcam**")

# Info singkat
st.info("📹 Klik **START** untuk memulai deteksi | 🟢 Hijau = Mask | 🔴 Merah = No Mask")

# Download model dengan multiple fallback methods
@st.cache_data(show_spinner=False)
def download_model_optimized():
    """Download model dengan multiple fallback untuk reliability"""
    model_path = 'mask_detector_svm.pkl'

    # Jika model sudah ada dan valid, skip download
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:  # Minimal 1MB
        return True

    file_id = "1z6CFkbbyDFsuMLPMHEQ9pI-zlM1x13s4"

    # Method 1: Direct download dengan requests
    try:
        with st.spinner("🔄 Downloading model... (Method 1/3)"):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            return True
    except Exception as e:
        st.warning(f"Method 1 failed: {str(e)[:50]}...")

    # Method 2: gdown library
    try:
        with st.spinner("🔄 Downloading model... (Method 2/3)"):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=True)

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            return True
    except Exception as e:
        st.warning(f"Method 2 failed: {str(e)[:50]}...")

    # Method 3: Alternative URL format
    try:
        with st.spinner("🔄 Downloading model... (Method 3/3)"):
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            return True
    except Exception as e:
        st.error(f"All download methods failed. Last error: {str(e)[:100]}")

    return False

# Load model dengan robust error handling
@st.cache_resource(show_spinner=False)
def load_model_optimized():
    """Load model dengan robust error handling dan fallback"""
    model_path = 'mask_detector_svm.pkl'

    # Cek apakah file model ada dan valid
    if not os.path.exists(model_path):
        st.info("📥 Model tidak ditemukan, mencoba download...")
        if not download_model_optimized():
            st.error("❌ Download gagal. Silakan upload manual.")
            return None

    # Verifikasi ukuran file
    if os.path.getsize(model_path) < 1000000:  # Kurang dari 1MB = corrupt
        st.warning("⚠️ File model corrupt, mencoba download ulang...")
        os.remove(model_path)
        if not download_model_optimized():
            st.error("❌ Re-download gagal. Silakan upload manual.")
            return None

    try:
        # Load model dengan error handling
        with st.spinner("🤖 Loading AI model..."):
            model = joblib.load(model_path)

        # Verifikasi model valid
        if hasattr(model, 'predict') and hasattr(model, 'decision_function'):
            st.success("✅ Model berhasil dimuat dan siap digunakan!")
            return model
        else:
            st.error("❌ Model format tidak valid")
            return None

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error loading model: {error_msg[:100]}")

        # Jika error karena versi scikit-learn, berikan saran
        if "version" in error_msg.lower():
            st.warning("⚠️ Kemungkinan masalah versi scikit-learn. Coba upload model yang kompatibel.")

        return None

@st.cache_resource
def load_face_cascade():
    """Load Haar Cascade untuk deteksi wajah"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("❌ Face cascade tidak dapat dimuat!")
            return None
        return face_cascade
    except Exception as e:
        st.error(f"❌ Error loading face cascade: {e}")
        return None

# Class untuk video processor (sama seperti detect_mask_webcam.py)
class MaskDetectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model_optimized()
        self.face_cascade = load_face_cascade()
        self.IMG_SIZE = 100
        self.frame_count = 0
        self.process_every_n_frames = 5  # Increase untuk mengurangi lag
        self.last_faces = []  # Cache hasil deteksi terakhir

        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.model is None or self.face_cascade is None:
            cv2.putText(img, "Model tidak tersedia", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "Upload file mask_detector_svm.pkl", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.frame_count += 1
        self.fps_frame_count += 1

        # Hitung FPS setiap detik (sama seperti desktop)
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
            self.fps_frame_count = 0

        # Proses deteksi dengan optimasi untuk mengurangi lag
        if self.frame_count % self.process_every_n_frames == 0:
            # Resize frame lebih kecil untuk deteksi super cepat
            small_frame = cv2.resize(img, (240, 180))  # Lebih kecil dari sebelumnya
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah dengan parameter yang lebih cepat
            faces_small = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.2,  # Lebih besar = lebih cepat
                minNeighbors=3,   # Lebih kecil = lebih cepat
                minSize=(20, 20), # Lebih kecil = lebih cepat
                flags=cv2.CASCADE_SCALE_IMAGE  # Optimasi tambahan
            )

            # Scale kembali koordinat ke ukuran asli
            scale_x = img.shape[1] / 240
            scale_y = img.shape[0] / 180

            faces = []
            for (x, y, w, h) in faces_small:
                faces.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                ))

            self.last_faces = faces
        else:
            # Gunakan hasil deteksi sebelumnya
            faces = self.last_faces

        # Proses prediksi untuk setiap wajah (sama seperti desktop)
        for (x, y, w, h) in faces:
            try:
                # Extract dan resize wajah
                face = img[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (self.IMG_SIZE, self.IMG_SIZE))
                face_flat = face_resized.flatten().reshape(1, -1)

                # Prediksi
                prediction = self.model.predict(face_flat)[0]
                confidence = abs(self.model.decision_function(face_flat)[0])

                # Label dan warna (sama seperti desktop)
                if prediction == 0:
                    label_text = "😷 Mask"
                    color = (0, 255, 0)  # Hijau
                else:
                    label_text = "😐 No Mask"
                    color = (0, 0, 255)  # Merah

                # Gambar rectangle dan text (sama seperti desktop)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

                # Background untuk text agar lebih jelas
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(img, (x, y-35), (x + text_size[0], y), color, -1)

                # Text label
                cv2.putText(img, label_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Confidence score
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(img, conf_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                continue

        # Tambahkan info FPS dan status (sama seperti desktop)
        cv2.putText(img, f"FPS: {self.current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit, 's' for screenshot", (10, img.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Konfigurasi RTC untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Main app (dengan optimasi performa)
def main():
    # Load model dengan optimasi (akan auto-download jika belum ada)
    model = load_model_optimized()

    if model is None:
        st.error("🚨 **Model tidak dapat dimuat!**")

        # Debug info
        model_path = 'mask_detector_svm.pkl'
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            st.info(f"📁 File ditemukan: {file_size:,} bytes")
            if file_size < 1000000:
                st.warning("⚠️ File terlalu kecil - kemungkinan corrupt")
        else:
            st.info("📁 File model tidak ditemukan")

        # Status dan solusi
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔄 Auto-Download")
            if st.button("🚀 Download Model", type="primary"):
                # Clear cache dan coba download
                st.cache_data.clear()
                st.cache_resource.clear()

                # Hapus file corrupt jika ada
                if os.path.exists(model_path):
                    os.remove(model_path)

                with st.spinner("Downloading model... Please wait"):
                    time.sleep(1)  # Give time for file deletion
                    st.rerun()

            st.info("💡 Model akan didownload otomatis (~472MB)")
            st.caption("Proses download mungkin memakan waktu 1-2 menit")

        with col2:
            st.markdown("### 📤 Manual Upload")
            uploaded_model = st.file_uploader(
                "Upload mask_detector_svm.pkl",
                type=['pkl'],
                help="Backup jika auto-download gagal"
            )

            if uploaded_model is not None:
                try:
                    # Hapus file lama jika ada
                    if os.path.exists(model_path):
                        os.remove(model_path)

                    # Save file baru
                    with open(model_path, 'wb') as f:
                        f.write(uploaded_model.getbuffer())

                    file_size = os.path.getsize(model_path)
                    st.success(f"✅ Model uploaded! ({file_size:,} bytes)")

                    if st.button("🔄 Load Model", type="primary"):
                        st.cache_resource.clear()
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")

        # Download link sebagai backup
        st.markdown("### 🔗 Manual Download")
        st.markdown("[📁 Download dari Google Drive](https://drive.google.com/file/d/1z6CFkbbyDFsuMLPMHEQ9pI-zlM1x13s4/view?usp=sharing)")
        st.caption("Jika auto-download gagal, download manual lalu upload di atas")

        # Tips untuk mengurangi lag
        st.markdown("### ⚡ Tips untuk Performa Optimal:")
        st.info("""
        - Gunakan koneksi internet yang stabil untuk download
        - Tutup aplikasi lain yang menggunakan kamera
        - Refresh halaman jika terasa lag
        - Model akan ter-cache setelah berhasil dimuat
        """)

    else:
        # WebRTC streamer dengan optimasi untuk mengurangi lag
        webrtc_ctx = webrtc_streamer(
            key="mask-detector",
            video_processor_factory=MaskDetectorProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},    # Resolusi optimal
                    "height": {"ideal": 480},   # Tidak terlalu tinggi
                    "frameRate": {"ideal": 15}  # FPS lebih rendah = less lag
                },
                "audio": False
            },
            async_processing=True,
        )

        # Status
        if webrtc_ctx.video_processor:
            st.success("🟢 Kamera aktif - Deteksi berjalan")
        else:
            st.info("👆 Klik **START** untuk memulai deteksi")

if __name__ == "__main__":
    main()
