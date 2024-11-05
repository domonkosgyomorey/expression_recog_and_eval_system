from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from imgsolver import imgsolver as isol
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)
imSolver = isol.ImgSolver("../../models", verbose=True)

@app.route('/evaluate', methods=['POST'])
def process_image():
    # Ellenőrizzük, hogy a fájl helyesen érkezett-e
    if 'image' not in request.files:
        return jsonify({'message': 'No image part in the request', 'status': 'error'})

    file = request.files['image']
    
    # Olvassuk be a fájlt
    try:
        file_data = file.read()
        img_data = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image decoding failed.")

        expression, result = imSolver.eval(image)

        return jsonify({
            'message': {'expression': expression, 'result': str(result)},
            'status': 'success'
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': str(e), 'status': 'error'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
