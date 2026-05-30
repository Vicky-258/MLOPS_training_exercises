from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    speaking_speed = float(request.form["speaking_speed"])
    filler_words = float(request.form["filler_words"])
    pause_count = float(request.form["pause_count"])
    eye_contact = float(request.form["eye_contact"])
    tone_variation = float(request.form["tone_variation"])

    # Create input array
    features = np.array([[
        speaking_speed,
        filler_words,
        pause_count,
        eye_contact,
        tone_variation
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0]

    # Decode prediction
    classes = ["Average", "Excellent", "Poor"]

    result = classes[prediction]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Presentation Category: {result}"
    )

if __name__ == "__main__":
    app.run(debug=True)