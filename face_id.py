import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading

class FaceID:
    def __init__(self, model="mobilefacenet.tflite", db="faces_db"):
        self.db = db
        os.makedirs(db, exist_ok=True)
        self.interpreter = tflite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]["index"]
        self.out = self.interpreter.get_output_details()[0]["index"]
        self.lock = threading.Lock()  # 添加线程锁
        self.load_db()

    def embed(self, face):
        face = cv2.resize(face, (112, 112))
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, 0)
        self.interpreter.set_tensor(self.inp, face)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out)[0].copy()

    def load_db(self):
        self.embs = {}
        for name in os.listdir(self.db):
            folder = os.path.join(self.db, name)
            if not os.path.isdir(folder):
                continue
            embs_list = []
            for f in os.listdir(folder):
                img_path = os.path.join(folder, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    embs_list.append(self.embed(img))
                except:
                    continue
            if embs_list:
                self.embs[name] = np.mean(np.array(embs_list), axis=0)
        if self.embs:
            self.names = list(self.embs.keys())
            self.emb_array = np.array(list(self.embs.values()))
        else:
            self.names = []
            self.emb_array = None

    def recognize(self, face, threshold=0.55):
        with self.lock:  # 使用线程锁
            if self.emb_array is None or len(self.names) == 0:
                return "Unknown", 1.0
            try:
                e = self.embed(face)
                d = np.linalg.norm(self.emb_array - e, axis=1)
                i = np.argmin(d)
                # 添加边界检查
                if i >= len(self.names) or i >= len(d):
                    return "Unknown", 1.0
                if d[i] < threshold:
                    return self.names[i], d[i]
                return "Unknown", d[i]
            except Exception as e:
                print(f"[ERROR] Recognition failed: {e}")
                return "Unknown", 1.0

    def add(self, name, face):
        with self.lock:  # 使用线程锁
            folder = os.path.join(self.db, name)
            os.makedirs(folder, exist_ok=True)
            idx = len(os.listdir(folder)) + 1
            img_path = os.path.join(folder, f"{idx}.jpg")
            cv2.imwrite(img_path, face)
            self.load_db()
            print(f"[INFO] Added face for {name} at {img_path}")
    
    def get_user_count(self):
        """获取用户数量"""
        return len(self.names) if self.names else 0
    
    def user_exists(self, username):
        """检查用户是否存在"""
        return username in self.names if self.names else False
