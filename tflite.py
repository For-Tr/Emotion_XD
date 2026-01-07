import tensorflow as tf
from keras.models import model_from_json
import numpy as np

# -------------------------
# Step 1: Load Keras model
# -------------------------
with open("fer.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("fer.weights.h5")
print("✅ Keras model loaded")

# -------------------------
# Step 2: Prepare TFLite converter
# -------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 仅使用 Pi 支持的内置 ops，避免生成新版本 FULLY_CONNECTED
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 启用优化（量化）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 强制 float16，确保 op 版本兼容 Pi 4
converter.target_spec.supported_types = [tf.float16]

# -------------------------
# Step 3: Convert to TFLite
# -------------------------
tflite_model = converter.convert()
print("✅ TFLite model converted")

# -------------------------
# Step 4: Save model
# -------------------------
tflite_path = "fer_pi4.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Pi-compatible TFLite model saved as '{tflite_path}'")

# -------------------------
# Optional: Test model input/output
# -------------------------
# Load interpreter to check
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Interpreter loaded successfully")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])
