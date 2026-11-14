"""RF-DETR segmentation inference with explicit model and class metadata."""

import json
from pathlib import Path

import cv2
import numpy as np


def label_detections(detections, class_names, class_id_offset=0, background_class_id=None):
    """Map checkpoint logit slots into contiguous ROS class IDs."""
    if background_class_id is not None:
        detections = detections[detections.class_id != background_class_id]
    ids = detections.class_id
    if ids is not None:
        ids = ids - class_id_offset
        if np.any(ids < 0) or np.any(ids >= len(class_names)):
            raise ValueError("Prediction class IDs do not match the supplied class names")
        detections.class_id = ids
        detections.data["class_name"] = np.asarray(
            [class_names[int(i)] for i in ids], dtype=object
        )
    return detections


class RFDetrBackend:
    """Keep RGB conversion and model preprocessing together."""

    def __init__(self, model_path, model_type=None, resolution=None, device="cuda"):
        from rfdetr import RFDETRSegNano, RFDETRSegSmall

        metadata_path = Path(model_path).with_suffix(".classes.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        else:
            labels = json.loads((Path(model_path).parent / "classes.json").read_text())
            if model_type is None:
                raise ValueError("Specify the model type when loading a raw classes.json mapping")
            metadata = {"class_names": ([labels[str(i)] for i in range(1, len(labels))]
                                        if isinstance(labels, dict) else labels[1:]),
                        "class_id_offset": 1, "background_class_id": 0}
        self.model_type = model_type or metadata.get("model_type", "rfdetr-seg-small")
        models = {"rfdetr-seg-small": (RFDETRSegSmall, 384),
                  "rfdetr-seg-nano": (RFDETRSegNano, 312)}
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        model_class, default_resolution = models[self.model_type]
        self.resolution = resolution or metadata.get("resolution", default_resolution)
        self.class_names = metadata["class_names"]
        self.class_id_offset = metadata["class_id_offset"]
        self.background_class_id = metadata["background_class_id"]
        self.model = model_class(
            pretrain_weights=str(Path(model_path).resolve()),
            resolution=self.resolution,
            num_classes=len(self.class_names),
            device=device,
        )
        self.model.model.class_names = (["background"] * self.class_id_offset) + self.class_names

    def predict(self, bgr, threshold=0.25):
        # predict() normalizes, resizes to model resolution and returns coordinates/masks
        # in the original image dimensions. Do not resize or normalize twice.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(rgb, threshold=threshold)
        return label_detections(
            detections, self.class_names, self.class_id_offset, self.background_class_id
        )


class RFDetrTRTBackend:
    """Use RF-DETR's TensorRT runtime and its native segmentation decoder."""

    def __init__(self, model_path):
        import torch
        from rfdetr.export._tensorrt.inference import TRTInference
        from rfdetr.models.postprocess import PostProcess

        metadata = json.loads(Path(model_path).with_suffix(".json").read_text())
        if metadata["model_type"] not in ("rfdetr-seg-small", "rfdetr-seg-nano"):
            raise ValueError("Expected an RF-DETR Seg Small or Nano engine")
        self.resolution = int(metadata["resolution"])
        if self.resolution <= 0:
            raise ValueError("Invalid engine resolution")
        self.class_names = metadata["class_names"]
        self.class_id_offset = metadata["class_id_offset"]
        self.background_class_id = metadata["background_class_id"]
        self.model = TRTInference(str(model_path), device="cuda", sync_mode=True)
        self.postprocess = PostProcess(num_select=metadata["num_select"])
        if self.model.input_names != ["input"]:
            raise ValueError("Expected one TensorRT image input named 'input'")
        binding = self.model.bindings["input"]
        if tuple(binding.shape) != (1, 3, self.resolution, self.resolution):
            raise ValueError("TensorRT input shape does not match metadata resolution")
        self.input_dtype = torch.from_numpy(np.empty(0, dtype=binding.dtype)).dtype
        device = self.model.engine_device
        self.pixel_scale = torch.tensor(255, device=device, dtype=torch.float32)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(3, 1, 1)
        self.image_shape = None
        self.target_sizes = None

    def predict(self, bgr, threshold=0.25):
        import supervision as sv
        import torch
        from torchvision.transforms import functional as F

        with torch.inference_mode():
            device = self.model.engine_device
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
            tensor = tensor.to(torch.float32).div_(self.pixel_scale)
            tensor = F.resize(tensor, [self.resolution, self.resolution], antialias=False)
            tensor.sub_(self.mean).div_(self.std)
            tensor = tensor.unsqueeze(0).to(self.input_dtype).contiguous()
            # execute_v2 uses TensorRT's synchronous execution path. Ensure the
            # input is ready even when preprocessing ran on a non-default stream.
            torch.cuda.current_stream(device).synchronize()
            outputs = self.model({"input": tensor})
            if self.image_shape != bgr.shape[:2]:
                self.image_shape = bgr.shape[:2]
                self.target_sizes = torch.tensor([self.image_shape], device=device)
            results = self.postprocess(
                {"pred_boxes": outputs["dets"], "pred_logits": outputs["labels"],
                 "pred_masks": outputs["masks"]},
                target_sizes=self.target_sizes,
                score_threshold=threshold,
            )[0]
            # Segmentation PostProcess already filters every output by this threshold.
            detections = sv.Detections(
                xyxy=results["boxes"].float().cpu().numpy(),
                confidence=results["scores"].float().cpu().numpy(),
                class_id=results["labels"].cpu().numpy(),
                mask=results["masks"].squeeze(1).cpu().numpy(),
            )
        return label_detections(
            detections, self.class_names, self.class_id_offset, self.background_class_id
        )


def load_backend(model_path):
    """Select checkpoint or locally exported TensorRT engine by suffix."""
    if Path(model_path).suffix.lower() in (".trt", ".engine"):
        return RFDetrTRTBackend(model_path)
    return RFDetrBackend(model_path)
