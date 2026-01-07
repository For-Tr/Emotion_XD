import cv2
import numpy as np
import os
from keras.models import model_from_json

# 几何对齐函数
def align_and_crop_face(gray, face_box):
    x, y, w, h = face_box

    # FER 需要额头 + 嘴
    y1 = max(0, int(y - 0.15 * h))
    y2 = min(gray.shape[0], int(y + 1.05 * h))

    x1 = max(0, int(x - 0.10 * w))
    x2 = min(gray.shape[1], int(x + 1.10 * w))

    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # 强制成正方形（FER统计结构）
    h2, w2 = face.shape
    size = min(h2, w2)
    y_start = (h2 - size) // 2
    x_start = (w2 - size) // 2
    face = face[y_start:y_start + size, x_start:x_start + size]

    # resize 到 48×48
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    return face

# 主类
class OptimizedEmotionDetector:
    def __init__(self):
        # 情绪类别
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # 加载 Keras 模型
        self.load_model()
        # 加载人脸检测器
        self.setup_detectors()

        # 加载训练集全局均值和标准差
        self.TRAIN_MEAN = np.load("train_mean.npy")
        self.TRAIN_STD = np.load("train_std.npy")

    # Load Keras model
    def load_model(self):
        with open("fer.json", "r") as f:
            self.emotion_model = model_from_json(f.read())
        self.emotion_model.load_weights("fer.weights.h5")
        print("✅ Keras CNN model loaded")

    # Face detectors
    def setup_detectors(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        prototxt = os.path.join(BASE_DIR, "deploy.prototxt")
        model = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(model):
            print("❌ Face detector files not found")
            exit(1)

        self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Face detectors loaded")

    # Face detection
    def detect_faces(self, frame):
        faces = []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        if len(faces) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            haar_faces = self.haar_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
            faces = haar_faces

        return faces

    # Emotion prediction
    def predict_emotion(self, face_roi):
        # 灰度图已传入
        face_roi = face_roi.astype("float32")
        face_roi -= self.TRAIN_MEAN
        face_roi /= (self.TRAIN_STD + 1e-6)

        # 调整维度 (batch, h, w, channel)
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        probs = self.emotion_model.predict(face_roi, verbose=0)[0]
        return probs

    # Main loop
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            faces = self.detect_faces(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for (x, y, w, h) in faces:
                # 对齐 + 裁剪
                face_roi = align_and_crop_face(gray, (x, y, w, h))
                if face_roi is None:
                    continue

                preds = self.predict_emotion(face_roi)
                idx = np.argmax(preds)
                conf = preds[idx] * 100

                # 绘制最高情绪
                label = f"{self.emotions[idx]}: {conf:.1f}%"
                color = (0, 255, 0) if conf > 70 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 绘制所有情绪占比（右侧）
                for i, e in enumerate(self.emotions):
                    sub_label = f"{e}:{preds[i] * 100:.1f}%"
                    # 横向位置在人脸框右侧，纵向按 i*15 排列
                    cv2.putText(frame, sub_label, (x + w + 10, y + i * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("FER CNN Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = OptimizedEmotionDetector()
    detector.run()
