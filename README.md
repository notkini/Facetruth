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

Open http://127.0.0.1:5000. The model weights are included in `mesonet_model.pth`.

## Deploy to Vercel

1. Push this repository to GitHub and import it in Vercel, or install the Vercel CLI and run `vercel`.
2. Keep the project root as the Root Directory. Vercel detects the Flask app in `app.py` and installs the runtime packages from `requirements.txt`.
3. Deploy. No environment variables or login provider are required.

The app uses CPU-only PyTorch wheels to keep the serverless bundle smaller. Vercel Functions are configured for up to 60 seconds; uploads are capped at 4 MB. Uploaded images are processed in memory and are not saved to disk.

## Notes

- The model accepts image uploads in JPG, PNG and WEBP formats.
- The model score reflects the classifier output and can be wrong; it is not forensic proof.
- `render.yaml` remains available for the existing Render deployment path.
