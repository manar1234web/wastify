from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from skimage.feature import hog
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

def load_model():
    for fname in ["svm_model.joblib", "tree_model.joblib"]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            return joblib.load(fpath), fname
    return None, None

model, model_name = load_model()
print("النموذج المحمّل:", model_name)

def extract_features(im: Image.Image) -> np.ndarray:
    rgb = im.convert("RGB").resize((128, 128))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    gray = np.asarray(rgb.convert("L").resize((64, 64)), dtype=np.float32) / 255.0

    h = np.histogram(hsv[..., 0], bins=18, range=(0, 256))[0]
    s = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0]
    v = np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0]
    colour = np.concatenate([h, s, v]) / (128.0 * 128.0)

    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([colour, shape]).astype(np.float32)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400
    if model is None:
        return jsonify({"error": "Model file (.joblib) not found"}), 500

    file = request.files["image"]
    with Image.open(file.stream) as im:
        feats = extract_features(im).reshape(1, -1)

    pred = int(model.predict(feats)[0])
    is_recyclable = (pred == 1)

    return jsonify({
        "model": model_name,
        "pred": pred,
        "type": "Recyclable" if is_recyclable else "Organic",
        "label_ar": "قابل لإعادة التدوير" if is_recyclable else "نفايات عضوية",
        "label_en": "Recyclable" if is_recyclable else "Organic Waste",
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )