"""Export the notebook checkpoint for the static GitHub Pages demo."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from voc_detector import load_detector

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
folder = ROOT / "site" / "model"
folder.mkdir(exist_ok=True)
export = folder / "detector.onnx"
model = load_detector(args.checkpoint)
torch.onnx.export(model, torch.rand(1, 3, 640, 640), export, opset_version=17,
                  dynamo=False, input_names=["image"], output_names=["boxes", "labels", "scores"])
parts = []
with export.open("rb") as source:
    while chunk := source.read(60 * 1024 * 1024):
        name = f"part-{len(parts)}.bin"
        (folder / name).write_bytes(chunk)
        parts.append({"file":name, "bytes":len(chunk), "sha256":hashlib.sha256(chunk).hexdigest()})
saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
(folder / "classes.json").write_text(json.dumps(saved["classes"]))
(folder / "export.json").write_text(json.dumps({"input":[1,3,640,640], "parts":parts}, indent=2))
export.unlink()
print("Exported", len(parts), "model parts to", folder)
