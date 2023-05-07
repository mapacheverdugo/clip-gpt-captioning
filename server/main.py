from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def root():
    return jsonify({
        'version': '0.0.1',
        'available': True,
        'message': 'API funcionando correctamente'
    })


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({
        'version': '0.0.1',
        'available': True,
        'message': 'API funcionando correctamente'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
