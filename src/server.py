import os

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

from predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = './data/upload'


@app.route('/', methods=['GET', 'POST'])
def root_route():
    return jsonify({
        'version': '0.0.1',
        'available': True,
        'message': 'API funcionando correctamente'
    })


@app.route('/predict/', methods=['POST'])
def predict_route():
    file = request.files.get('file')

    print(file)

    if file.filename != '':
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        caption = predict(file_path, size='L', plot=False)

        return jsonify({
            'filename': file.filename,
            'path': file_path,
            'caption': caption
        })

    return jsonify({
        'version': '0.0.1',
        'available': True,
        'message': 'API funcionando correctamente'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
