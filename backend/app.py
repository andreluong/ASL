from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app, origins="*")

def decode_image(data_url: str) -> Image.Image:
    # Handle both raw base64 and data URLs (data:image/jpeg;base64,...)
    if ',' in data_url:
        encoded = data_url.split(',')[1]
    else:
        encoded = data_url
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data keys:", data.keys())

    if 'image' in data:
        image = decode_image(data['image'])
        print("Single frame, size:", image.size)
        
        # TODO: model
        prediction, confidence = 'A', 0.88

    elif 'frames' in data:
        raw_frames = data['frames']
        print("Frames type:", type(raw_frames[0]))

        # Flatten in case frames are nested lists
        flat_frames = [f if isinstance(f, str) else f[0] for f in raw_frames]
        frames = [decode_image(f) for f in flat_frames]
        print(f"Frame buffer count: {len(frames)}, size: {frames[0].size}")
        
        # TODO: model

        prediction, confidence = 'C', 0.55

    else:
        return jsonify({'error': 'No image or frames provided'}), 400

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
