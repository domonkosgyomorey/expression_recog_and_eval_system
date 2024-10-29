from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from imgsolver import imgsolver as isol
import matplotlib.pylab as plt

app = Flask(__name__)
CORS(app)
imSolver = isol.ImgSolver("../../models")

@app.route('/evaluate', methods=['POST'])
def process_image():

    file = request.files['image']
    file_data = file.read()
    img_data = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if image is None:
        print("Error during decoding the image.")
        return jsonify({
            'message': 'Decoding was unsuccessful',
            'status': 'error'
        })
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    expression, result = imSolver.eval(image)
    
    return jsonify({
        'message': {'expression': expression, 'result': str(result)},
        'status': 'success'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)