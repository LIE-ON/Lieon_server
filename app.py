from flask import Flask, request, jsonify
import os
import inference
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Use the environment variable for temporary directory or default to "/tmp"
    temp_dir = os.getenv("TEMP", "/tmp")
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file temporarily
        file.save(file_path)
        logger.info(f"File saved to {file_path}")

        # Update the model and scaler paths as needed
        example_model_path = "svm_model.pkl"
        example_scaler_path = "scaler.pkl"

        # Call the inference function (assuming it's correctly implemented in your 'inference' module)
        result = inference.inference(file_path, example_model_path, example_scaler_path)

        # Ensure the result is a Python native type (e.g., int) and not a numpy type
        result = int(result)  # Convert numpy.int64 to a native Python int

        # Prepare the result as a JSON response and return it
        response = {
            "result": result,  # Now 'result' is a native Python int
        }

        # Return the response with the correct Content-Type header
        return jsonify(response)

    except Exception as e:
        # Log the error for debugging purposes
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    finally:
        # Clean up the uploaded file after processing, even if an error occurs
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"File {file_path} deleted after processing.")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")

if __name__ == '__main__':
    # Run the Flask app on all interfaces (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
