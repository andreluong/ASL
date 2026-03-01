from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # TODO: Add model and pass image to it
    
    
    # TODO: replace
    return jsonify({
        'prediction': 'A',
        'confidence': 0.88
    })
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
