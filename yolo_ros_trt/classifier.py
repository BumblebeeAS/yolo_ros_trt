"""ROI classification with preprocessing and class names embedded in the export."""

import json
from pathlib import Path

import cv2
import numpy as np


class ExportedClassifier:
    def __init__(self, model_path, device="cpu"):
        self.is_trt = Path(model_path).suffix.lower() in (".trt", ".engine")
        if self.is_trt:
            from yolo_ros_trt.trt_engine import TensorRTEngine

            metadata = json.loads(Path(model_path).with_suffix(".json").read_text())
            if metadata["model_type"] not in ("buoy-color-pipeline", "classifier-pipeline"):
                raise ValueError("Expected an exported ROI classifier pipeline engine")
            contract = metadata["contract"]
            self.model = TensorRTEngine(model_path)
        else:
            import torch

            self.device = torch.device(device)
            if self.device.type == "cpu":
                torch.set_num_threads(2)
            extra = {"contract.json": ""}
            exported = torch.export.load(model_path, extra_files=extra)
            contract = json.loads(extra["contract.json"])
            self.model = exported.module().to(self.device)
        self.classes = contract["classes"]
        if (not isinstance(self.classes, list) or not self.classes
                or not all(isinstance(name, str) and name for name in self.classes)
                or len(set(self.classes)) != len(self.classes)):
            raise ValueError("Classifier contract must contain unique nonempty class names")
        self.image_shape = tuple(contract["image"]["shape"])
        if contract["image"]["color"] != "RGB" or contract["context_expansion"]:
            raise ValueError("Expected RGB input and zero-context cropping")

    def predict(self, bgr, bbox):
        if bgr.shape != self.image_shape or bgr.dtype != np.uint8:
            raise ValueError(f"Classifier expects uint8 BGR image of shape {self.image_shape}; got {bgr.shape}")
        x, y, width, height = bbox
        if not np.isfinite(bbox).all() or min(width, height) <= 0:
            return "unknown", 0.0
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = np.asarray([x - width/2, y - height/2, x + width/2, y + height/2], dtype=np.float32)
        if self.is_trt:
            result = self.model({"image": image, "box": box})
            probabilities, valid = result["probabilities"], result["valid"].item()
        else:
            import torch

            with torch.inference_mode():
                probabilities, valid = self.model(
                    torch.from_numpy(image).to(self.device), torch.from_numpy(box).to(self.device)
                )
        if not bool(valid):
            return "unknown", 0.0
        if self.is_trt:
            index = int(probabilities[0].argmax())
            return self.classes[index], float(probabilities[0, index])
        confidence, index = probabilities[0].max(dim=0)
        return self.classes[int(index)], float(confidence)
