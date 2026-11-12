#@title Website prediction code
"""Standalone inference for the VOC 2007 Faster R-CNN checkpoint."""
import torch
from PIL import ImageOps
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import to_tensor


CLASSES = (
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def restore_detector(saved, device="cpu"):
    if saved["format_version"] != 1 or tuple(saved["classes"]) != CLASSES:
        raise ValueError("Incompatible detector checkpoint or class order.")
    # The COCO-v1 builder uses FrozenBatchNorm and changes its epsilon.
    # Recreate both explicitly so loading never needs a pretrained download.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d
    )
    model = FasterRCNN(backbone, num_classes=len(CLASSES), **saved["model_kwargs"])
    for layer in model.modules():
        if isinstance(layer, FrozenBatchNorm2d):
            layer.eps = saved["frozen_bn_eps"]
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


def load_detector(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return restore_detector(saved, device)


@torch.inference_mode()
def predict_image(model, image, threshold=0.5):
    """Return JSON-ready detections in EXIF-corrected, original-image pixels."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    image = ImageOps.exif_transpose(image).convert("RGB")
    device = next(model.parameters()).device
    output = model([to_tensor(image).to(device)])[0]
    keep = output["scores"] >= threshold
    boxes = output["boxes"][keep].cpu().tolist()
    labels = output["labels"][keep].cpu().tolist()
    scores = output["scores"][keep].cpu().tolist()
    return {
        "width": image.width, "height": image.height, "box_format": "xyxy",
        "detections": [
            {"class_id": label, "label": CLASSES[label], "score": score, "box": box}
            for box, label, score in zip(boxes, labels, scores)
        ],
    }
