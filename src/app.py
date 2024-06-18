from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import io
import time

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cnn_weights_path = os.path.join(BASE_DIR, 'models', 'model_cnn.h5')
mlp_weights_path = os.path.join(BASE_DIR, 'models', 'model_linear.h5')

model_cnn = load_model(cnn_weights_path)
model_mlp = load_model(mlp_weights_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    try:
        image_stream = io.BytesIO(file.read())
        
        img = load_img(image_stream, target_size=(28, 28), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        start_time_cnn = time.time()
        prediction_cnn = model_cnn.predict(img_array)
        end_time_cnn = time.time()
        inference_time_cnn = end_time_cnn - start_time_cnn
        digit_cnn = np.argmax(prediction_cnn)

        return jsonify({
            'digit_cnn': int(digit_cnn),
            'inference_time_cnn': inference_time_cnn
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_mlp', methods=['POST'])
def predict_mlp():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    try:
        image_stream = io.BytesIO(file.read())
        
        img = load_img(image_stream, target_size=(28, 28), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        start_time_mlp = time.time()
        prediction_mlp = model_mlp.predict(img_array)
        end_time_mlp = time.time()
        inference_time_mlp = end_time_mlp - start_time_mlp
        digit_mlp = np.argmax(prediction_mlp)

        return jsonify({
            'digit_mlp': int(digit_mlp),
            'inference_time_mlp': inference_time_mlp
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
