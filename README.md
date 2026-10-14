# FaceTruth

FaceTruth is an open, login-free web app for checking a still image with a MesoNet deepfake classifier. The interface presents the model output as an estimate, not proof.

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Inference uses the included `mesonet_model.onnx` file.

If you update the PyTorch checkpoint, install `requirements-export.txt` and run
`python export_model_onnx.py` to regenerate the ONNX model.

## Deploy to Vercel

1. Push this repository to GitHub and import it in Vercel, or install the Vercel CLI and run `vercel`.
2. Keep the project root as the Root Directory. Vercel detects the Flask app in `app.py` and installs the runtime packages from `requirements.txt`.
3. Deploy. No environment variables or login provider are required.

The app uses ONNX Runtime for CPU inference to keep the serverless bundle within Vercel's function size limit. Vercel Functions are configured for up to 60 seconds; uploads are capped at 4 MB. Uploaded images are processed in memory and are not saved to disk.

## Notes

- The model accepts image uploads in JPG, PNG and WEBP formats.
- The model score reflects the classifier output and can be wrong; it is not forensic proof.
- `render.yaml` remains available for the existing Render deployment path.
