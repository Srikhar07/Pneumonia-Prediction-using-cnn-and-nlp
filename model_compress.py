import tensorflow as tf
from keras.models import load_model

# 1️⃣ Load the .h5 model safely
model = load_model(
    r"D:\ML PROJECTS\Multi-Modal Medical Assistant for Pneumonia Prediction using Chest X-rays and Symptom Analysis\model\lung.h5",
    compile=False
)

print("Model loaded successfully")

# 2️⃣ Convert to TFLite (compressed)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3️⃣ Save the smaller model
with open("lung_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully")



