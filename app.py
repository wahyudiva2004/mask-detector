import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="🎭 Mask Detector",
    page_icon="😷",
    layout="centered"
)

# Judul aplikasi
st.title("🎭 Real-Time Mask Detector")
st.markdown("**Deteksi penggunaan masker secara real-time**")

# Info singkat
st.info("📹 Klik **START** untuk memulai deteksi | 🟢 Hijau = Mask | 🔴 Merah = No Mask")

# Load model (simple version)
@st.cache_resource
def load_model():
    """Load model SVM untuk deteksi mask"""
    model_path = 'mask_detector_svm.pkl'

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("✅ Model berhasil dimuat!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        st.error("❌ Model tidak ditemukan! Upload file `mask_detector_svm.pkl`")
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
        self.model = load_model()
        self.face_cascade = load_face_cascade()
        self.IMG_SIZE = 100
        self.frame_count = 0
        self.process_every_n_frames = 3  # Sama seperti desktop app
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

        # Proses deteksi hanya setiap N frame untuk performa (sama seperti desktop)
        if self.frame_count % self.process_every_n_frames == 0:
            # Resize frame untuk deteksi lebih cepat
            small_frame = cv2.resize(img, (320, 240))
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah di frame kecil
            faces_small = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.1,
                minNeighbors=5,  # Sama seperti desktop
                minSize=(30, 30)  # Sama seperti desktop
            )

            # Scale kembali koordinat ke ukuran asli
            scale_x = img.shape[1] / 320
            scale_y = img.shape[0] / 240

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

# Main app (simple version)
def main():
    # Load model
    model = load_model()

    if model is None:
        st.error("🚨 **Model tidak ditemukan!**")
        st.markdown("### 📤 Upload Model")

        uploaded_model = st.file_uploader(
            "Upload file mask_detector_svm.pkl",
            type=['pkl'],
            help="Upload file model yang sudah didownload"
        )

        if uploaded_model is not None:
            try:
                # Save uploaded model
                with open('mask_detector_svm.pkl', 'wb') as f:
                    f.write(uploaded_model.getbuffer())
                st.success("✅ Model berhasil diupload!")
                st.info("🔄 Refresh halaman untuk menggunakan model")
                if st.button("🔄 Refresh Sekarang"):
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error upload model: {str(e)}")

        st.markdown("### 🔗 Download Model")
        st.markdown("[📁 Download dari Google Drive](https://drive.google.com/file/d/1z6CFkbbyDFsuMLPMHEQ9pI-zlM1x13s4/view?usp=sharing)")

    else:
        st.success("✅ Model siap digunakan!")

        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="mask-detector",
            video_processor_factory=MaskDetectorProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
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
