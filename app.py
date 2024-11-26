from flask import Flask, request, Response, jsonify
import os
import inference
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    temp_dir = os.getenv("TEMP", "/tmp")  # Temporary directory for saving uploaded files
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file temporarily
        file.save(file_path)

        # Call the inference method from the inference module
        example_model_path = "svm_model.pkl"  # Update as needed
        example_scaler_path = "scaler.pkl"   # Update as needed

        # Perform inference
        result = inference.inference(file_path, example_model_path, example_scaler_path)

        # Return result as JSON
        obj = {
            "result": np.int64(result),
        }
        return json.dumps(obj, default=np_encoder)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)