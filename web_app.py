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
import hashlib

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, session
from keras.models import model_from_json

# 导入原有的检测器和对齐函数
from run_emotion import OptimizedEmotionDetector, align_and_crop_face

# 导入人脸识别系统
try:
    from face_id_adapter import FaceID
    FACE_ID_AVAILABLE = True
except Exception as e:
    print(f"[!] 人脸识别系统不可用: {e}")
    FACE_ID_AVAILABLE = False

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
FACE_DB_PATH = 'database/faces_db'

# 加载模型
detector = None
face_recognizer = None

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            display_name TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            total_logins INTEGER DEFAULT 0,
            face_encodings_count INTEGER DEFAULT 0
        )
    ''')
    
    # 表情记录表（添加用户关联）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER,
            username TEXT DEFAULT 'guest',
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_probabilities TEXT,
            image_path TEXT,
            ai_analysis TEXT,
            session_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            user_id INTEGER,
            username TEXT DEFAULT 'guest',
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            total_detections INTEGER DEFAULT 0,
            dominant_emotion TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 数据库迁移：检查并添加缺失的列
    try:
        # 检查 emotion_records 表是否有 user_id 列
        cursor.execute("PRAGMA table_info(emotion_records)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'user_id' not in columns:
            print("[*] 迁移数据库：添加 user_id 列到 emotion_records 表")
            cursor.execute("ALTER TABLE emotion_records ADD COLUMN user_id INTEGER")
        
        if 'username' not in columns:
            print("[*] 迁移数据库：添加 username 列到 emotion_records 表")
            cursor.execute("ALTER TABLE emotion_records ADD COLUMN username TEXT DEFAULT 'guest'")
        
        # 检查 sessions 表是否有 user_id 列
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'user_id' not in columns:
            print("[*] 迁移数据库：添加 user_id 列到 sessions 表")
            cursor.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER")
        
        if 'username' not in columns:
            print("[*] 迁移数据库：添加 username 列到 sessions 表")
            cursor.execute("ALTER TABLE sessions ADD COLUMN username TEXT DEFAULT 'guest'")
        
        print("[✓] 数据库迁移完成")
    except Exception as e:
        print(f"[!] 数据库迁移出错: {e}")
    
    conn.commit()
    conn.close()
    
    # 创建人脸数据库目录
    os.makedirs(FACE_DB_PATH, exist_ok=True)

def load_detector():
    """加载表情检测器"""
    global detector
    if detector is None:
        print("[*] 正在加载表情识别模型...")
        detector = OptimizedEmotionDetector()
        print("[✓] 模型加载成功！")
    return detector

def load_face_recognizer():
    """加载人脸识别器"""
    global face_recognizer
    if face_recognizer is None and FACE_ID_AVAILABLE:
        try:
            print("[*] 正在加载人脸识别系统...")
            face_recognizer = FaceID(model="mobilefacenet.tflite", db=FACE_DB_PATH)
            print("[✓] 人脸识别系统加载成功！")
        except Exception as e:
            print(f"[!] 人脸识别系统加载失败: {e}")
            print("[!] 提示：如果缺少 mobilefacenet.tflite 模型文件，人脸识别功能将不可用")
            face_recognizer = None
    return face_recognizer

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

def save_emotion_record(emotion, confidence, probabilities, image_path=None, ai_analysis=None, session_id=None, user_id=None, username='guest'):
    """保存表情记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO emotion_records 
        (emotion, confidence, all_probabilities, image_path, ai_analysis, session_id, user_id, username)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (emotion, confidence, json.dumps(probabilities), image_path, 
          json.dumps(ai_analysis) if ai_analysis else None, session_id, user_id, username))
    
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

# ==================== 用户管理函数 ====================

def create_user(username, display_name=None):
    """创建新用户"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, display_name)
            VALUES (?, ?)
        ''', (username, display_name or username))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "用户名已存在"
    except Exception as e:
        conn.close()
        return None, str(e)

def get_user_by_username(username):
    """根据用户名获取用户信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, display_name, last_login, total_logins, face_encodings_count FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            'id': row[0],
            'username': row[1],
            'display_name': row[2],
            'last_login': row[3],
            'total_logins': row[4],
            'face_encodings_count': row[5]
        }
    return None

def update_user_login(user_id):
    """更新用户登录信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users 
        SET last_login = ?, total_logins = total_logins + 1
        WHERE id = ?
    ''', (datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

def update_user_face_count(username, count):
    """更新用户的人脸照片数量"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users 
        SET face_encodings_count = ?
        WHERE username = ?
    ''', (count, username))
    conn.commit()
    conn.close()

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
        
        # 尝试识别用户（如果启用了人脸识别）
        user_id = None
        username = 'guest'
        recognized = False
        
        recognizer = load_face_recognizer()
        if recognizer is not None and face_coords is not None:
            try:
                x, y, w, h = face_coords
                face_img = image[y:y+h, x:x+w]
                recognized_name, distance = recognizer.recognize(face_img)
                
                if recognized_name != "Unknown":
                    user = get_user_by_username(recognized_name)
                    if user:
                        user_id = user['id']
                        username = user['username']
                        recognized = True
            except Exception as e:
                print(f"[!] 人脸识别失败: {e}")
        
        # AI 分析
        ai_analysis = get_ai_analysis(emotion, confidence)
        
        # 保存记录（关联用户信息）
        session_id = data.get('session_id', f'web_{int(time.time())}')
        record_id = save_emotion_record(emotion, confidence, probabilities, 
                                       ai_analysis=ai_analysis, session_id=session_id,
                                       user_id=user_id, username=username)
        
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
            'timestamp': datetime.now().isoformat(),
            'user': {
                'recognized': recognized,
                'username': username,
                'user_id': user_id,
                'is_guest': username == 'guest'
            }
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
        user_filter = request.args.get('user', None)  # 支持按用户过滤
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 构建查询
        if user_filter:
            if user_filter == 'current' and 'user_id' in session:
                # 只显示当前登录用户的记录
                cursor.execute('''
                    SELECT id, timestamp, emotion, confidence, all_probabilities, ai_analysis, user_id, username
                    FROM emotion_records
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (session['user_id'], limit))
            elif user_filter == 'guest':
                # 只显示访客记录
                cursor.execute('''
                    SELECT id, timestamp, emotion, confidence, all_probabilities, ai_analysis, user_id, username
                    FROM emotion_records
                    WHERE username = 'guest' OR user_id IS NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            else:
                # 按指定用户名过滤
                cursor.execute('''
                    SELECT id, timestamp, emotion, confidence, all_probabilities, ai_analysis, user_id, username
                    FROM emotion_records
                    WHERE username = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_filter, limit))
        else:
            # 显示所有记录
            cursor.execute('''
                SELECT id, timestamp, emotion, confidence, all_probabilities, ai_analysis, user_id, username
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
                'ai_analysis': json.loads(row[5]) if row[5] else None,
                'user_id': row[6],
                'username': row[7] or 'guest'
            })
        
        conn.close()
        
        return jsonify({'records': records})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """获取统计数据，支持按用户过滤"""
    try:
        user_filter = request.args.get('user', None)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 构建WHERE子句
        where_clause = ""
        params = []
        if user_filter:
            if user_filter == 'current' and 'user_id' in session:
                where_clause = "WHERE user_id = ?"
                params = [session['user_id']]
            elif user_filter == 'guest':
                where_clause = "WHERE username = 'guest' OR user_id IS NULL"
            else:
                where_clause = "WHERE username = ?"
                params = [user_filter]
        
        # 总记录数
        cursor.execute(f"SELECT COUNT(*) FROM emotion_records {where_clause}", params)
        total_records = cursor.fetchone()[0]
        
        # 按表情分组统计
        cursor.execute(f'''
            SELECT emotion, COUNT(*) as count 
            FROM emotion_records 
            {where_clause}
            GROUP BY emotion
        ''', params)
        emotion_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 平均置信度
        cursor.execute(f"SELECT AVG(confidence) FROM emotion_records {where_clause}", params)
        avg_confidence = cursor.fetchone()[0] or 0
        
        # 最近24小时的记录数
        cursor.execute(f'''
            SELECT COUNT(*) FROM emotion_records 
            {where_clause}
            {"AND" if where_clause else "WHERE"} datetime(timestamp) > datetime('now', '-1 day')
        ''', params)
        records_24h = cursor.fetchone()[0]
        
        # 按用户统计（如果没有指定用户过滤）
        user_stats = []
        if not user_filter:
            cursor.execute('''
                SELECT username, COUNT(*) as count 
                FROM emotion_records 
                GROUP BY username
                ORDER BY count DESC
            ''')
            user_stats = [{'username': row[0] or 'guest', 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total_records': total_records,
            'emotion_counts': emotion_counts,
            'avg_confidence': round(avg_confidence, 3),
            'records_24h': records_24h,
            'user_stats': user_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def api_export():
    """导出记录为CSV，支持按用户过滤"""
    try:
        import csv
        from io import StringIO
        
        user_filter = request.args.get('user', None)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 构建WHERE子句和参数
        where_clause = ""
        params = []
        if user_filter:
            if user_filter == 'current' and 'user_id' in session:
                where_clause = "WHERE user_id = ?"
                params = [session['user_id']]
            elif user_filter == 'guest':
                where_clause = "WHERE username = 'guest' OR user_id IS NULL"
            else:
                where_clause = "WHERE username = ?"
                params = [user_filter]
        
        query = f'''
            SELECT timestamp, emotion, confidence, session_id, username
            FROM emotion_records
            {where_clause}
            ORDER BY timestamp DESC
        '''
        
        cursor.execute(query, params)
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['时间戳', '表情', '置信度', '会话ID', '用户'])
        
        for row in cursor.fetchall():
            writer.writerow([
                row[0],
                EMOTIONS_CN[EMOTIONS.index(row[1])],
                f"{row[2]:.2%}",
                row[3] or 'N/A',
                row[4] or 'guest'
            ])
        
        conn.close()
        
        output.seek(0)
        
        # 根据过滤器生成文件名
        filename = 'emotion_records'
        if user_filter == 'current':
            filename += '_my'
        elif user_filter == 'guest':
            filename += '_guest'
        elif user_filter:
            filename += f'_{user_filter}'
        filename += '.csv'
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
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

@app.route('/user')
def user():
    """用户中心页面"""
    template_path = 'templates/user.html'
    if os.path.exists(template_path):
        return render_template('user.html')
    else:
        return "<h1>用户中心</h1><p>模板加载中...</p>"

# ==================== 用户认证 API ====================

@app.route('/api/register_face', methods=['POST'])
def api_register_face():
    """用户人脸注册 API - 支持多张照片"""
    try:
        recognizer = load_face_recognizer()
        if recognizer is None:
            return jsonify({'error': '人脸识别系统未加载'}), 503
        
        data = request.get_json()
        
        if 'username' not in data:
            return jsonify({'error': '未提供用户名'}), 400
        
        if 'images' not in data or not data['images']:
            return jsonify({'error': '未提供人脸图片'}), 400
        
        username = data['username']
        display_name = data.get('display_name', username)
        images = data['images']  # 多张图片的 base64 数据
        
        # 检查用户是否已存在
        existing_user = get_user_by_username(username)
        if existing_user:
            return jsonify({'error': '用户名已存在'}), 400
        
        # 创建用户记录
        user_id, error = create_user(username, display_name)
        if error:
            return jsonify({'error': error}), 400
        
        # 保存多张人脸照片
        success_count = 0
        for img_data in images:
            try:
                # 解码图片
                image_data = img_data.split(',')[1] if ',' in img_data else img_data
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # 检测人脸
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) == 0:
                    continue
                
                # 取第一个检测到的人脸
                x, y, w, h = faces[0]
                face_img = image[y:y+h, x:x+w]
                
                # 添加到人脸识别系统
                recognizer.add(username, face_img)
                success_count += 1
                
            except Exception as e:
                print(f"[!] 处理人脸图片失败: {e}")
                continue
        
        if success_count == 0:
            return jsonify({'error': '未能成功保存任何人脸图片'}), 400
        
        # 更新用户的人脸照片数量
        update_user_face_count(username, success_count)
        
        return jsonify({
            'success': True,
            'message': f'用户 {username} 注册成功',
            'user_id': user_id,
            'face_count': success_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login_face', methods=['POST'])
def api_login_face():
    """人脸登录 API"""
    try:
        recognizer = load_face_recognizer()
        if recognizer is None:
            return jsonify({'error': '人脸识别系统未加载'}), 503
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': '未提供人脸图片'}), 400
        
        # 解码图片
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无效的图片数据'}), 400
        
        # 检测人脸
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'error': '未检测到人脸'}), 404
        
        # 取第一个检测到的人脸
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        
        # 识别人脸
        username, distance = recognizer.recognize(face_img)
        
        if username == "Unknown":
            return jsonify({
                'success': False,
                'message': '未识别到注册用户',
                'username': 'guest'
            })
        
        # 获取用户信息
        user = get_user_by_username(username)
        if not user:
            return jsonify({'error': '用户数据不一致'}), 500
        
        # 更新登录信息
        update_user_login(user['id'])
        
        # 设置 session
        session['user_id'] = user['id']
        session['username'] = user['username']
        
        return jsonify({
            'success': True,
            'message': f'欢迎回来，{user["display_name"]}！',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'display_name': user['display_name'],
                'total_logins': user['total_logins'] + 1
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize_face', methods=['POST'])
def api_recognize_face():
    """识别人脸（不登录，仅识别）"""
    try:
        recognizer = load_face_recognizer()
        if recognizer is None:
            return jsonify({'error': '人脸识别系统未加载'}), 503
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': '未提供人脸图片'}), 400
        
        # 解码图片
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无效的图片数据'}), 400
        
        # 检测人脸
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'error': '未检测到人脸'}), 404
        
        # 取第一个检测到的人脸
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        
        # 识别人脸
        username, distance = recognizer.recognize(face_img)
        
        if username == "Unknown":
            return jsonify({
                'success': True,
                'recognized': False,
                'username': 'guest',
                'message': '未识别到注册用户'
            })
        
        # 获取用户信息
        user = get_user_by_username(username)
        
        return jsonify({
            'success': True,
            'recognized': True,
            'username': user['username'] if user else username,
            'display_name': user['display_name'] if user else username,
            'user_id': user['id'] if user else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """用户登出"""
    session.clear()
    return jsonify({'success': True, 'message': '已登出'})

@app.route('/api/current_user', methods=['GET'])
def api_current_user():
    """获取当前登录用户"""
    if 'user_id' in session:
        user = get_user_by_username(session['username'])
        if user:
            return jsonify({
                'logged_in': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'display_name': user['display_name'],
                    'total_logins': user['total_logins']
                }
            })
    
    return jsonify({'logged_in': False, 'user': None})

if __name__ == '__main__':
    # 初始化数据库
    init_db()
    
    # 预加载模型
    print("[*] 预加载模型...")
    load_detector()
    load_face_recognizer()
    
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