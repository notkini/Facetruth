from pathlib import Path

import torch

from model import MesoNet


ROOT = Path(__file__).resolve().parent
model = MesoNet().cpu()
model.load_state_dict(
    torch.load(ROOT / "mesonet_model.pth", map_location="cpu", weights_only=True)
)
model.eval()

torch.onnx.export(
    model,
    torch.zeros(1, 3, 256, 256),
    ROOT / "mesonet_model.onnx",
    input_names=["image"],
    output_names=["probability"],
    opset_version=17,
)
