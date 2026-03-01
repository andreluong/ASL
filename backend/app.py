from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp

app = Flask(__name__)
CORS(app, origins="*")

# Load model
checkpoint = torch.load("ml/asl_model.pth", map_location="cpu")
GESTURES = checkpoint["gestures"]
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, len(GESTURES))
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def image_to_skeleton(pil_image):
    img = np.array(pil_image.convert("RGB"))
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        results = hands.process(img)
        if not results.multi_hand_landmarks:
            return None
        
        canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                canvas, hand_landmarks, HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
        
        canvas = cv2.resize(canvas, (224, 224))
        return canvas



@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    skeleton = image_to_skeleton(pil_image)
    if skeleton is None:
        return jsonify({'prediction': None, 'confidence': 0, 'error': 'No hand detected'})

    tensor = transform(skeleton).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = probs.max(1)
    
    return jsonify({
        'prediction': GESTURES[idx.item()],
        'confidence': round(conf.item(), 3)
    })
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
