from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)

# Define 14 class labels
class_labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Tuberculosis", "covid"
]

# Load the CheXNet model
def load_chexnet_model(weights_path):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(class_labels), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(weights_path)
    return model

# Path to your weights file
weights_path = "model/CheXNet weights.h5"
model = load_chexnet_model(weights_path)
print("Model loaded successfully!")

# Preprocess the uploaded image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(file_path)

    try:
        # Preprocess the image
        image_array = preprocess_image(file_path)
        
        # Predict using the model
        predictions = model.predict(image_array)[0]
        result = {class_labels[i]: round(predictions[i] * 100, 2) for i in range(len(class_labels))}
        
        # Return the predictions as JSON
        return render_template('result.html', predictions=result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

    

if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
