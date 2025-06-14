import cv2
import joblib
import time
import os
import sys

def main():
    # Cek apakah model ada
    model_path = 'mask_detector_svm.pkl'
    if not os.path.exists(model_path):
        print("❌ Error: File model tidak ditemukan!")
        print("💡 Solusi:")
        print("1. Download model dari Google Drive")
        print("2. Atau jalankan training: python train_mask_detector.py")
        return

    try:
        print("📥 Loading model...")
        model = joblib.load(model_path)
        print("✅ Model berhasil dimuat!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load face cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Error: Face cascade tidak dapat dimuat!")
            return
        print("✅ Face cascade berhasil dimuat!")
    except Exception as e:
        print(f"❌ Error loading face cascade: {e}")
        return

    IMG_SIZE = 100

    # Inisialisasi kamera dengan optimasi
    print("📹 Menginisialisasi kamera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Tidak dapat mengakses kamera!")
        print("💡 Pastikan kamera tidak digunakan aplikasi lain")
        return

    # Optimasi kamera untuk performa
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer untuk real-time

    print("✅ Kamera berhasil diinisialisasi!")
    print("🎯 Tekan 'q' untuk keluar, 's' untuk screenshot")

    # Variables untuk optimasi
    frame_count = 0
    process_every_n_frames = 3  # Proses setiap 3 frame untuk performa
    last_faces = []  # Cache hasil deteksi terakhir

    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Tidak dapat membaca frame dari kamera")
                break

            frame_count += 1
            fps_frame_count += 1

            # Hitung FPS setiap detik
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Proses deteksi hanya setiap N frame untuk performa
            if frame_count % process_every_n_frames == 0:
                # Resize frame untuk deteksi lebih cepat
                small_frame = cv2.resize(frame, (320, 240))
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                # Deteksi wajah di frame kecil
                faces_small = face_cascade.detectMultiScale(
                    gray_small,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Scale kembali koordinat ke ukuran asli
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240

                faces = []
                for (x, y, w, h) in faces_small:
                    faces.append((
                        int(x * scale_x),
                        int(y * scale_y),
                        int(w * scale_x),
                        int(h * scale_y)
                    ))

                last_faces = faces
            else:
                # Gunakan hasil deteksi sebelumnya
                faces = last_faces

            # Proses prediksi untuk setiap wajah
            for (x, y, w, h) in faces:
                try:
                    # Extract dan resize wajah
                    face = frame[y:y+h, x:x+w]
                    if face.size == 0:
                        continue

                    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face_flat = face_resized.flatten().reshape(1, -1)

                    # Prediksi
                    prediction = model.predict(face_flat)[0]
                    confidence = abs(model.decision_function(face_flat)[0])

                    # Label dan warna
                    if prediction == 0:
                        label_text = "😷 Mask"
                        color = (0, 255, 0)  # Hijau
                    else:
                        label_text = "😐 No Mask"
                        color = (0, 0, 255)  # Merah

                    # Gambar rectangle dan text
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                    # Background untuk text agar lebih jelas
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x, y-35), (x + text_size[0], y), color, -1)

                    # Text label
                    cv2.putText(frame, label_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Confidence score
                    conf_text = f"Conf: {confidence:.2f}"
                    cv2.putText(frame, conf_text, (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")
                    continue

            # Tambahkan info FPS dan status
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Tampilkan frame
            try:
                cv2.imshow("🎭 Real-Time Mask Detection", frame)
            except cv2.error as e:
                print(f"❌ Error menampilkan frame: {e}")
                print("💡 Install opencv-python (bukan opencv-python-headless)")
                break

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Keluar dari aplikasi...")
                break
            elif key == ord('s'):
                # Screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"mask_detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot disimpan: {filename}")

    except KeyboardInterrupt:
        print("\n👋 Aplikasi dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error tidak terduga: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Kamera dan jendela ditutup")

if __name__ == "__main__":
    main()
