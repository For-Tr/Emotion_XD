import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Face alignment
def align_and_crop_face(gray, face_box):
    x, y, w, h = face_box

    y1 = max(0, int(y - 0.15 * h))
    y2 = min(gray.shape[0], int(y + 1.05 * h))

    x1 = max(0, int(x - 0.10 * w))
    x2 = min(gray.shape[1], int(x + 1.10 * w))

    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None

    h2, w2 = face.shape
    size = min(h2, w2)
    y0 = (h2 - size) // 2
    x0 = (w2 - size) // 2
    face = face[y0:y0+size, x0:x0+size]

    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    return face


# Emotion Detector (Pi 4)
class EmotionDetectorPi:
    def __init__(self):
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # Haar face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load normalization
        self.TRAIN_MEAN = np.load("train_mean.npy")
        self.TRAIN_STD = np.load("train_std.npy")

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path="fer.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("Pi 4 Emotion system ready")

    # Face detect
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        return faces, gray

    # Emotion predict
    def predict(self, face):
        face = face.astype("float32")
        face -= self.TRAIN_MEAN
        face /= (self.TRAIN_STD + 1e-6)

        face = face.reshape(1, 48, 48, 1)

        self.interpreter.set_tensor(self.input_details[0]['index'], face)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return preds

    # Main loop
    def run(self):
        cap = cv2.VideoCapture(0)

        # 必须降低分辨率，否则 Pi 扛不住
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not cap.isOpened():
            print("Camera not available")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_roi = align_and_crop_face(gray, (x, y, w, h))
                if face_roi is None:
                    continue

                preds = self.predict(face_roi)
                idx = np.argmax(preds)
                conf = preds[idx] * 100

                label = f"{self.emotions[idx]} {conf:.1f}%"
                color = (0,255,0) if conf > 70 else (0,255,255)

                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # All emotions
                for i, e in enumerate(self.emotions):
                    txt = f"{e}:{preds[i]*100:.1f}%"
                    cv2.putText(frame, txt, (x+w+5, y+15*i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            cv2.imshow("Raspberry Pi Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    EmotionDetectorPi().run()
