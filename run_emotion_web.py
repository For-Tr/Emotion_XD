from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from picamera2 import Picamera2
from run_emotion_pi4 import EmotionDetectorPi, align_and_crop_face  # 原来的类和方法

# ---------------------------
# Flask Web Streaming
# ---------------------------
app = Flask(__name__)
detector = EmotionDetectorPi()

# 初始化 Pi Camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(preview_config)
picam2.start()

# 网页模板
HTML_PAGE = """
<html>
<head><title>Pi Emotion Detection</title></head>
<body>
<h1>Raspberry Pi 4 Emotion Detection</h1>
<img src="{{ url_for('video_feed') }}">
<p>按 Ctrl+C 停止服务器</p>
</body>
</html>
"""

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)
        faces, gray = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = align_and_crop_face(gray, (x, y, w, h))
            if face_roi is None:
                continue

            preds = detector.predict(face_roi)
            idx = np.argmax(preds)
            conf = preds[idx]*100
            label = f"{detector.emotions[idx]} {conf:.1f}%"
            color = (0,255,0) if conf>70 else (0,255,255)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            # 显示所有情绪百分比
            for i, e in enumerate(detector.emotions):
                txt = f"{e}:{preds[i]*100:.1f}%"
                cv2.putText(frame, txt, (x+w+5, y+15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

        # 转 JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Pi 4 Emotion Web system ready")
    app.run(host='0.0.0.0', port=5000, debug=False)
