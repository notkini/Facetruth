from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from model import MesoNet

if os.environ.get("VERCEL"):
    logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

device = torch.device("cpu")
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
model = MesoNet().to(device)
model.load_state_dict(
    torch.load(ROOT / "mesonet_model.pth", map_location=device, weights_only=True)
)
model.eval()


def predict(image_bytes: bytes) -> tuple[float, str]:
    with Image.open(io.BytesIO(image_bytes)) as source:
        if source.format not in ALLOWED_FORMATS:
            raise ValueError("Please upload a JPG, PNG or WEBP image.")
        image = source.convert("RGB")
        preview = io.BytesIO()
        image.save(preview, format="JPEG", quality=88)

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = float(model(tensor).reshape(-1)[0].item())
    preview_data = base64.b64encode(preview.getvalue()).decode("ascii")
    return probability, f"data:image/jpeg;base64,{preview_data}"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_image():
    if request.method == "GET":
        return render_template("predict.html")

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return render_template("predict.html", error="Choose an image to run a check."), 400

    image_bytes = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return (
            render_template(
                "predict.html", error="That image is too large. Choose a file under 4 MB."
            ),
            413,
        )

    try:
        probability, result_image = predict(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        return (
            render_template(
                "predict.html", error="We couldn’t read that image. Try a JPG, PNG or WEBP file."
            ),
            400,
        )
    except Exception:
        app.logger.exception("Image analysis failed")
        return (
            render_template(
                "predict.html", error="The image could not be analyzed. Please try again."
            ),
            500,
        )

    # The single sigmoid output is the model's score for the real class.
    prob_real = probability * 100
    prob_fake = (1 - probability) * 100
    return render_template(
        "predict.html",
        result_image=result_image,
        prediction="Real" if probability > 0.5 else "Fake",
        filename=secure_filename(uploaded.filename) or "uploaded-image",
        prob_real=f"{prob_real:.1f}",
        prob_fake=f"{prob_fake:.1f}",
    )


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return (
        render_template("predict.html", error="That image is too large. Choose a file under 4 MB."),
        413,
    )


@app.route("/ping")
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
