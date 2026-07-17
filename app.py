from flask import Flask, render_template, request
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from werkzeug.utils import secure_filename
from model import MesoNet

# ------------------ APP SETUP ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------ MODEL LOAD ------------------
device = torch.device("cpu")
model = MesoNet().to(device)

try:
    model.load_state_dict(torch.load("mesonet_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise Exception("STOP: Fix model loading")

# ------------------ PREDICTION ------------------
def predict(image_path):
    try:
        # IMPORTANT: Properly close image after use
        with Image.open(image_path) as img:
            image = img.convert('RGB')

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = outputs.cpu().detach().numpy()

        # Handle different model outputs
        if probs.shape[-1] == 2:
            prob = probs[0][1]
        else:
            prob = probs[0][0]

        return float(prob)

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# ------------------ ROUTES ------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('predict.html', error="No file selected")

        file = request.files['file']

        if file.filename == '':
            return render_template('predict.html', error="No file selected")

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            prob = predict(file_path)

            if prob is None:
                return render_template('predict.html', error="Prediction failed")

            prob_real = prob * 100
            prob_fake = (1 - prob) * 100
            prediction_label = 'Real' if prob > 0.5 else 'Fake'

            # ------------------ GRAPH ------------------
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(['Real', 'Fake'], [prob_real, prob_fake])
            ax.set_ylabel('Confidence (%)')
            ax.set_ylim(0, 100)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            buffer.seek(0)

            graph_string = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # ❌ NO FILE DELETE (fix for WinError 32)

            return render_template(
                'predict.html',
                graph=graph_string,
                prediction=prediction_label,
                filename=filename,
                prob_real=f"{prob_real:.1f}",
                prob_fake=f"{prob_fake:.1f}"
            )

        except Exception as e:
            print(f"Error: {e}")
            return render_template('predict.html', error="Something went wrong")

    return render_template('predict.html')

@app.route('/ping')
def ping():
    return {'status': 'ok'}

# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)