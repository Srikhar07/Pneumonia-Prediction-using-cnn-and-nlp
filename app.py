import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request
from keras.preprocessing import image

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "lung_model.tflite")
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "model", "symptom_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD TFLITE MODEL ----------------
interpreter = tf.lite.Interpreter(model_path=CNN_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD SYMPTOM MODELS ----------------
symptom_model = joblib.load(SYMPTOM_MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# -------- IMAGE MODEL --------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64,64))  
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    score = float(output[0][0])

    return score


# -------- SYMPTOM MODEL --------
def predict_symptoms(text):
    vec = tfidf.transform([text])
    prob = symptom_model.predict_proba(vec)[0][1]
    return float(prob)


# ---------------- ROUTES ----------------
@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # -------- FORM DATA --------
        name = request.form.get('user_name')
        age = request.form.get('age')
        phone_number = request.form.get('phone')
        email = request.form.get('email')
        symptoms = request.form.get('symptoms')

        img_file = request.files.get('image')

        if img_file is None or img_file.filename == "":
            return "No image uploaded", 400

        # -------- SAVE IMAGE --------
        img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
        img_file.save(img_path)

        # -------- GET SCORES --------
        image_score = predict_image(img_path)
        symptom_score = predict_symptoms(symptoms)

        # -------- FINAL COMBINED CONFIDENCE --------
        final_score = (0.5 * image_score) + (0.5 * symptom_score)
        # -------- FINAL PREDICTION LABEL --------
        if final_score >= 0.5:
            final_prediction = "PNEUMONIA DETECTED"
        else:
            final_prediction = "NORMAL"

        return render_template(
            'result.html',
            name=name,
            age=age,
            phone_number=phone_number,
            email=email,
            symptoms=symptoms,
            image_path='uploads/' + img_file.filename,
            image_conf=image_score * 100,
            symptom_conf=symptom_score * 100,
            final_conf=final_score * 100,
            final_prediction=final_prediction
        )

    except Exception as e:
        print("Prediction error:", e)
        return "Internal Server Error", 500


# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    app.run(port=8080, debug=True)

