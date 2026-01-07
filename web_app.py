#!/usr/bin/env python3
"""
面部表情识别管理平台 - 基于已训练的 FER 模型
Facial Expression Recognition Management Platform
"""

import os
import sys
import json
import time
import base64
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from keras.models import model_from_json

# 导入原有的检测器和对齐函数
from run_emotion import OptimizedEmotionDetector, align_and_crop_face

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fer-emotion-recognition-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# 全局变量
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTIONS_CN = ["愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶", "中性"]
EMOTION_COLORS = ['#e74c3c', '#9b59b6', '#34495e', '#f39c12', '#3498db', '#1abc9c', '#95a5a6']

DB_PATH = 'database/emotions.db'

# 加载模型
detector = None

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_probabilities TEXT,
            image_path TEXT,
            ai_analysis TEXT,
            session_id TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            total_detections INTEGER DEFAULT 0,
            dominant_emotion TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def load_detector():
    """加载表情检测器"""
    global detector
    if detector is None:
        print("[*] 正在加载表情识别模型...")
        detector = OptimizedEmotionDetector()
        print("[✓] 模型加载成功！")
    return detector

def predict_emotion(image):
    """
    预测表情
    image: BGR 彩色图像
    返回: emotion, confidence, probabilities, face_coords
    """
    det = load_detector()
    
    # 检测人脸
    faces = det.detect_faces(image)
    
    if len(faces) == 0:
        return None, None, None, None
    
    # 使用最大的人脸
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = face
    
    # 转灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 对齐和裁剪
    face_roi = align_and_crop_face(gray, face)
    
    if face_roi is None:
        return None, None, None, None
    
    # 预测
    probabilities = det.predict_emotion(face_roi)
    
    # 获取最高概率的情绪
    emotion_idx = np.argmax(probabilities)
    emotion = EMOTIONS[emotion_idx]
    confidence = float(probabilities[emotion_idx])
    
    return emotion, confidence, probabilities.tolist(), face

def get_ai_analysis(emotion, confidence):
    """AI 情感分析"""
    emotion_cn = EMOTIONS_CN[EMOTIONS.index(emotion)]
    
    suggestions = {
        'angry': {
            'analysis': '您当前的表情显示出愤怒情绪。愤怒是一种强烈的情绪反应，通常源于受挫、不公或压力。',
            'suggestions': ['尝试深呼吸，给自己几分钟冷静时间', '通过运动释放负面能量', '与信任的朋友倾诉', '写下让你愤怒的原因，理性分析'],
            'wellness_score': max(0, 50 - confidence * 30)
        },
        'disgust': {
            'analysis': '您的表情反映出厌恶感。这可能是对某些刺激的自然反应。',
            'suggestions': ['远离让你感到不适的环境', '保持积极的心态', '寻找让你感到舒适的活动', '尝试换个角度看问题'],
            'wellness_score': max(0, 60 - confidence * 25)
        },
        'fear': {
            'analysis': '您的表情显示出恐惧或担忧。适度的恐惧是自我保护机制，但过度焦虑会影响生活质量。',
            'suggestions': ['识别并面对恐惧的来源', '练习正念冥想和放松技巧', '与专业人士交流', '建立安全感和自信心'],
            'wellness_score': max(0, 55 - confidence * 28)
        },
        'happy': {
            'analysis': '太棒了！您的笑容表明您心情愉悦。保持这种积极的状态。',
            'suggestions': ['分享你的快乐给周围的人', '记录下美好的时刻', '保持感恩的心态', '继续做让你开心的事情'],
            'wellness_score': min(100, 70 + confidence * 30)
        },
        'sad': {
            'analysis': '您的表情显示出悲伤情绪。悲伤是正常的情感体验，允许自己感受它。',
            'suggestions': ['给自己时间处理情绪', '与亲友交流，寻求支持', '进行户外活动，接触大自然', '写日记记录感受'],
            'wellness_score': max(0, 55 - confidence * 30)
        },
        'surprise': {
            'analysis': '您的表情显示出惊讶。这是对意外刺激的自然反应。',
            'suggestions': ['给自己时间处理新信息', '保持开放和好奇的心态', '理性评估突发情况', '享受生活中的意外惊喜'],
            'wellness_score': 75
        },
        'neutral': {
            'analysis': '您的表情处于中性状态，显示出平静和稳定。',
            'suggestions': ['保持内心的平和', '培养积极的兴趣爱好', '定期与朋友社交', '尝试新的体验增添生活色彩'],
            'wellness_score': 70
        }
    }
    
    emotion_info = suggestions.get(emotion, suggestions['neutral'])
    
    return {
        'emotion': emotion,
        'emotion_cn': emotion_cn,
        'confidence': confidence,
        'analysis': emotion_info['analysis'],
        'suggestions': emotion_info['suggestions'],
        'wellness_score': emotion_info['wellness_score'],
        'timestamp': datetime.now().isoformat()
    }

def save_emotion_record(emotion, confidence, probabilities, image_path=None, ai_analysis=None, session_id=None):
    """保存表情记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO emotion_records 
        (emotion, confidence, all_probabilities, image_path, ai_analysis, session_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (emotion, confidence, json.dumps(probabilities), image_path, 
          json.dumps(ai_analysis) if ai_analysis else None, session_id))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def get_emotion_statistics(days=7):
    """获取统计数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    cursor.execute('''
        SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM emotion_records
        WHERE timestamp > ?
        GROUP BY emotion
        ORDER BY count DESC
    ''', (since_date,))
    
    stats = cursor.fetchall()
    
    cursor.execute('''
        SELECT DATE(timestamp) as date, emotion, COUNT(*) as count
        FROM emotion_records
        WHERE timestamp > ?
        GROUP BY DATE(timestamp), emotion
        ORDER BY date DESC
    ''', (since_date,))
    
    daily_trends = cursor.fetchall()
    
    conn.close()
    
    return {
        'overall': [{'emotion': row[0], 'count': row[1], 'avg_confidence': row[2]} for row in stats],
        'daily_trends': [{'date': row[0], 'emotion': row[1], 'count': row[2]} for row in daily_trends]
    }

# ==================== 路由定义 ====================

@app.route('/')
def index():
    # 创建简单的HTML模板路径检查
    template_path = 'templates/index.html'
    if os.path.exists(template_path):
        return render_template('index.html')
    else:
        return "<h1>面部表情识别管理平台</h1><p>模板文件正在加载...</p><p>请访问 /api/predict POST 接口进行测试</p>"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': '未提供图片数据'}), 400
        
        # 解码 base64 图片
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无效的图片数据'}), 400
        
        # 预测表情
        emotion, confidence, probabilities, face_coords = predict_emotion(image)
        
        if emotion is None:
            return jsonify({'error': '未检测到人脸'}), 404
        
        # AI 分析
        ai_analysis = get_ai_analysis(emotion, confidence)
        
        # 保存记录
        session_id = data.get('session_id', f'web_{int(time.time())}')
        record_id = save_emotion_record(emotion, confidence, probabilities, 
                                       ai_analysis=ai_analysis, session_id=session_id)
        
        # 准备响应
        response = {
            'success': True,
            'record_id': record_id,
            'emotion': emotion,
            'emotion_cn': EMOTIONS_CN[EMOTIONS.index(emotion)],
            'confidence': confidence,
            'probabilities': {
                EMOTIONS[i]: {
                    'name': EMOTIONS_CN[i],
                    'value': float(probabilities[i]),
                    'color': EMOTION_COLORS[i]
                }
                for i in range(len(EMOTIONS))
            },
            'face_detected': True,
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[!] 预测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未提供文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 保存文件
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取图片
        image = cv2.imread(filepath)
        
        if image is None:
            return jsonify({'error': '无法读取图片'}), 400
        
        # 预测表情
        emotion, confidence, probabilities, face_coords = predict_emotion(image)
        
        if emotion is None:
            return jsonify({'error': '未检测到人脸'}), 404
        
        # AI 分析
        ai_analysis = get_ai_analysis(emotion, confidence)
        
        # 保存记录
        record_id = save_emotion_record(emotion, confidence, probabilities, 
                                       image_path=filepath, ai_analysis=ai_analysis)
        
        # 准备响应
        response = {
            'success': True,
            'record_id': record_id,
            'emotion': emotion,
            'emotion_cn': EMOTIONS_CN[EMOTIONS.index(emotion)],
            'confidence': confidence,
            'probabilities': {
                EMOTIONS[i]: {
                    'name': EMOTIONS_CN[i],
                    'value': float(probabilities[i]),
                    'color': EMOTION_COLORS[i]
                }
                for i in range(len(EMOTIONS))
            },
            'ai_analysis': ai_analysis,
            'image_path': filepath,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[!] 上传错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def api_statistics():
    try:
        days = request.args.get('days', 7, type=int)
        stats = get_emotion_statistics(days)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/records')
def api_records():
    try:
        limit = request.args.get('limit', 50, type=int)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, emotion, confidence, all_probabilities, ai_analysis
            FROM emotion_records
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        records = []
        for row in cursor.fetchall():
            records.append({
                'id': row[0],
                'timestamp': row[1],
                'emotion': row[2],
                'emotion_cn': EMOTIONS_CN[EMOTIONS.index(row[2])],
                'confidence': row[3],
                'probabilities': json.loads(row[4]) if row[4] else [],
                'ai_analysis': json.loads(row[5]) if row[5] else None
            })
        
        conn.close()
        
        return jsonify({'records': records})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def api_export():
    try:
        import csv
        from io import StringIO
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, emotion, confidence, session_id
            FROM emotion_records
            ORDER BY timestamp DESC
        ''')
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['时间戳', '表情', '置信度', '会话ID'])
        
        for row in cursor.fetchall():
            writer.writerow([
                row[0],
                EMOTIONS_CN[EMOTIONS.index(row[1])],
                f"{row[2]:.2%}",
                row[3] or 'N/A'
            ])
        
        conn.close()
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=emotion_records.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    template_path = 'templates/dashboard.html'
    if os.path.exists(template_path):
        return render_template('dashboard.html')
    else:
        return "<h1>数据仪表板</h1><p>模板加载中...</p>"

@app.route('/history')
def history():
    template_path = 'templates/history.html'
    if os.path.exists(template_path):
        return render_template('history.html')
    else:
        return "<h1>历史记录</h1><p>模板加载中...</p>"

if __name__ == '__main__':
    # 初始化数据库
    init_db()
    
    # 预加载模型
    print("[*] 预加载模型...")
    load_detector()
    
    print("\n" + "="*60)
    print("  面部表情识别管理平台")
    print("  基于训练好的 FER CNN 模型")
    print("="*60)
    print("\n[✓] 服务器启动成功！")
    print("[✓] 访问地址: http://localhost:5002")
    print("[✓] 数据仪表板: http://localhost:5002/dashboard")
    print("[✓] 历史记录: http://localhost:5002/history")
    print("\n[✓] 使用真实训练的深度学习模型")
    print("[✓] 模型文件: fer.weights.h5")
    print("\n按 Ctrl+C 停止服务器\n")
    
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)