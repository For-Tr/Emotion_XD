#!/usr/bin/env python3
"""
人脸识别系统诊断脚本
用于测试在树莓派上人脸识别系统是否正常工作
"""

import os
import sys

print("="*60)
print("人脸识别系统诊断")
print("="*60)

# 1. 检查模型文件
print("\n[1] 检查模型文件...")
model_path = "mobilefacenet.tflite"
if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    print(f"    ✓ 模型文件存在: {model_path} ({size/1024/1024:.2f} MB)")
else:
    print(f"    ✗ 模型文件不存在: {model_path}")
    print("    提示: 需要下载 mobilefacenet.tflite 文件")

# 2. 检查 tflite_runtime
print("\n[2] 检查 tflite_runtime...")
try:
    import tflite_runtime.interpreter as tflite
    print("    ✓ tflite_runtime 已安装 (树莓派推荐)")
    TFLITE_RUNTIME_OK = True
except ImportError:
    print("    ✗ tflite_runtime 未安装")
    print("    安装命令: pip install tflite-runtime")
    TFLITE_RUNTIME_OK = False

# 3. 检查 TensorFlow
print("\n[3] 检查 TensorFlow...")
try:
    import tensorflow as tf
    print(f"    ✓ TensorFlow 已安装 (版本: {tf.__version__})")
    TENSORFLOW_OK = True
except ImportError:
    print("    ✗ TensorFlow 未安装")
    TENSORFLOW_OK = False

# 4. 检查其他依赖
print("\n[4] 检查其他依赖...")
deps = {
    'cv2': 'opencv-python',
    'numpy': 'numpy'
}

for module, package in deps.items():
    try:
        __import__(module)
        print(f"    ✓ {package} 已安装")
    except ImportError:
        print(f"    ✗ {package} 未安装")

# 5. 尝试加载 FaceID
print("\n[5] 尝试加载 FaceID 类...")

if os.path.exists(model_path):
    # 尝试树莓派版本
    if TFLITE_RUNTIME_OK:
        try:
            print("    尝试加载 face_id.py (tflite_runtime)...")
            from face_id import FaceID
            print("    ✓ face_id.py 导入成功")
            
            try:
                face_recognizer = FaceID(model=model_path, db="database/faces_db")
                print("    ✓ FaceID 实例化成功")
                print(f"    ✓ 当前已注册用户数: {face_recognizer.get_user_count()}")
            except Exception as e:
                print(f"    ✗ FaceID 实例化失败: {e}")
        except Exception as e:
            print(f"    ✗ face_id.py 导入失败: {e}")
    
    # 尝试标准版本
    if TENSORFLOW_OK:
        try:
            print("    尝试加载 face_id_adapter.py (tensorflow)...")
            from face_id_adapter import FaceID as FaceIDAdapter
            print("    ✓ face_id_adapter.py 导入成功")
            
            try:
                face_recognizer = FaceIDAdapter(model=model_path, db="database/faces_db")
                print("    ✓ FaceID 实例化成功")
                print(f"    ✓ 当前已注册用户数: {face_recognizer.get_user_count()}")
            except Exception as e:
                print(f"    ✗ FaceID 实例化失败: {e}")
        except Exception as e:
            print(f"    ✗ face_id_adapter.py 导入失败: {e}")
else:
    print(f"    跳过测试（缺少模型文件: {model_path}）")

# 总结
print("\n" + "="*60)
print("诊断总结")
print("="*60)

if not os.path.exists(model_path):
    print("❌ 缺少模型文件 mobilefacenet.tflite")
    print("   解决方案: 下载模型文件到项目根目录")
elif not (TFLITE_RUNTIME_OK or TENSORFLOW_OK):
    print("❌ 缺少 TFLite 运行时")
    print("   解决方案（树莓派）: pip install tflite-runtime")
    print("   解决方案（其他平台）: pip install tensorflow")
else:
    print("✓ 环境配置正常，人脸识别系统应该可以工作")

print("\n" + "="*60)