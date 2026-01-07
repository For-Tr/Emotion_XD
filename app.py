from flask import Flask, Response, render_template_string, request
import cv2
import numpy as np
from picamera2 import Picamera2
from run_emotion_pi4 import EmotionDetectorPi, align_and_crop_face
from face_id import FaceID

app = Flask(__name__)
detector = EmotionDetectorPi()
faceid = FaceID()

picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(cfg)
picam2.start()

last_face = None
last_emotion_probs = None

HOME_HTML = """
<html>
<head>
<style>
body { font-family: Arial, sans-serif; text-align: center; background-color: #d0e7f9; margin: 0; padding: 0; }
h2 { color: #1a1a1a; margin-top: 20px; }
.video-container { display: inline-block; position: relative; margin-top: 20px; }
#video-feed { border: 4px solid #555; border-radius: 15px; width: 800px; height: 600px; }
form { margin-top: 25px; }
input[type=text] { padding: 6px; font-size: 16px; border-radius: 5px; border: 1px solid #888; }
button { padding: 6px 12px; font-size: 16px; border-radius: 5px; border: none; background-color: #2196F3; color: #fff; cursor: pointer; }
button:hover { background-color: #1976D2; }
</style>
</head>
<body>
<h2>Emotion + Face ID</h2>
<div class="video-container">
    <img id="video-feed" src="/video_feed">
</div>
<form action="/register" method="post">
    Name: <input name="name">
    <button type="submit">Register Face</button>
</form>
</body>
</html>
"""

RESULT_HTML = """
<html>
<body style="text-align:center; font-family: Arial; background-color: #d0e7f9;">
<h2>{{ message }}</h2>
<form action="/">
<button type="submit">Back to Camera</button>
</form>
</body>
</html>
"""


def gen():
    global last_face, last_emotion_probs
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces, gray_frame = detector.detect_faces(frame)
        last_emotion_probs = {}
        for (x, y, w, h) in faces:
            roi_gray = align_and_crop_face(gray_frame, (x, y, w, h))
            if roi_gray is None:
                continue
            roi_color = frame[y:y + h, x:x + w]
            roi_color_resized = cv2.resize(roi_color, (112, 112))
            last_face = roi_color_resized.copy()
            name, score = faceid.recognize(last_face, threshold=0.55)
            emo_input = cv2.resize(roi_gray, (48, 48))
            preds = detector.predict(emo_input)
            emo_idx = np.argmax(preds)
            emo = detector.emotions[emo_idx]
            emo_prob = float(preds[emo_idx])
            last_emotion_probs = {detector.emotions[i]: float(preds[i]) for i in range(len(detector.emotions))}

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 姓名显示为橙色，主要情绪及百分比显示在下方
            cv2.putText(frame, name, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)  # 橙色
            cv2.putText(frame, f"{emo}: {emo_prob * 100:.1f}%", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # 红色显示情绪和百分比

        ret, buf = cv2.imencode(".jpg", frame)
        frame_bytes = buf.tobytes()
        yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


@app.route("/")
def home():
    return render_template_string(HOME_HTML)


@app.route("/video_feed")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register", methods=["POST"])
def register():
    global last_face
    name = request.form.get("name", "").strip()
    if last_face is not None and name != "":
        faceid.add(name, last_face)
        message = f"Registration successful: {name} added!"
    else:
        message = "Registration failed: No face detected or invalid name."
    return render_template_string(RESULT_HTML, message=message)


@app.route("/emotion_probs")
def emotion_probs():
    global last_emotion_probs
    if last_emotion_probs is None:
        return {}
    return last_emotion_probs


if __name__ == "__main__":
    print("Pi 4 Emotion + Face ID Web system ready")
    app.run(host="0.0.0.0", port=5000, debug=False)
