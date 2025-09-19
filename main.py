from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_data = joblib.load("model.joblib")
pipeline = model_data["pipeline"]
target_names = model_data["target_names"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = pipeline.predict(features)
        class_name = target_names[prediction[0]]
        return jsonify({"prediction": int(prediction[0]), "class_name": class_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
