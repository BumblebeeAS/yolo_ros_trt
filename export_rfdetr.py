"""Export an RF-DETR segmentation checkpoint and its runtime metadata."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from yolo_ros_trt.rfdetr_backend import RFDetrBackend, RFDetrTRTBackend


def validate_export(reference, engine_path, image_paths, threshold):
    """Compare classes, scores, boxes and masks in original-image coordinates."""
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    engine = RFDetrTRTBackend(engine_path)
    report = {"samples": [], "detections": 0}
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read validation image: {path}")
        images = [image]
        if image.shape[:2] != (360, 640):
            images.append(cv2.resize(image, (640, 360)))
        for sample in images:
            # Warm the native model and device before measuring each image shape.
            reference.predict(sample, threshold)
            engine.predict(sample, threshold)
            start = time.perf_counter()
            expected = reference.predict(sample, threshold)
            cpu_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            actual = engine.predict(sample, threshold)
            trt_ms = (time.perf_counter() - start) * 1000
            if len(expected) != len(actual):
                raise RuntimeError(f"Detection count differs for {path}: {len(expected)} vs {len(actual)}")
            metrics = {"box_iou_min": 1.0, "mask_iou_min": 1.0, "score_error_max": 0.0}
            if len(expected):
                lo = np.maximum(expected.xyxy[:, None, :2], actual.xyxy[None, :, :2])
                hi = np.minimum(expected.xyxy[:, None, 2:], actual.xyxy[None, :, 2:])
                intersection = np.maximum(hi - lo, 0).prod(axis=2)
                areas_a = np.maximum(expected.xyxy[:, 2:] - expected.xyxy[:, :2], 0).prod(axis=1)
                areas_b = np.maximum(actual.xyxy[:, 2:] - actual.xyxy[:, :2], 0).prod(axis=1)
                iou = intersection / np.maximum(areas_a[:, None] + areas_b[None, :] - intersection, 1e-12)
                same_class = expected.class_id[:, None] == actual.class_id[None, :]
                rows, cols = linear_sum_assignment(np.where(same_class, -iou, 1e6))
                if not same_class[rows, cols].all():
                    raise RuntimeError(f"Class predictions differ for {path}")
                metrics["box_iou_min"] = float(iou[rows, cols].min())
                metrics["score_error_max"] = float(np.abs(expected.confidence[rows] - actual.confidence[cols]).max())
                mask_ious = []
                for a, b in zip(rows, cols):
                    union = np.logical_or(expected.mask[a], actual.mask[b]).sum()
                    mask_ious.append(float(np.logical_and(expected.mask[a], actual.mask[b]).sum() / union) if union else 1.0)
                metrics["mask_iou_min"] = min(mask_ious)
                if (metrics["box_iou_min"] < .99 or metrics["mask_iou_min"] < .98
                        or metrics["score_error_max"] > .01):
                    raise RuntimeError(f"TensorRT parity failed for {path}: {metrics}")
            report["detections"] += len(actual)
            report["samples"].append({"image": path, "shape": list(sample.shape),
                                      "classes": actual.data["class_name"].tolist(),
                                      "cpu_ms": cpu_ms, "trt_ms": trt_ms, **metrics})
    if not report["detections"]:
        raise RuntimeError("Validation images produced no detections; use visible targets")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=["rfdetr-seg-small", "rfdetr-seg-nano"])
    parser.add_argument("--resolution", type=int, help="Defaults to the model's native resolution")
    parser.add_argument("--validation-images", nargs="+", default=[])
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--fp16", action="store_true", help="Enable only after FP32 parity is checked")
    args = parser.parse_args()
    if (args.resolution is not None and args.resolution <= 0) or not 0 <= args.threshold <= 1:
        parser.error("Resolution must be positive and threshold must be between 0 and 1")
    import torch
    torch.set_num_threads(2)
    backend = RFDetrBackend(args.checkpoint, model_type=args.model_type,
                           resolution=args.resolution, device="cpu")
    metadata = {
        "model_type": backend.model_type,
        "resolution": backend.resolution,
        "class_names": backend.class_names,
        "class_id_offset": backend.class_id_offset,
        "background_class_id": backend.background_class_id,
        "num_select": backend.model.model.postprocess.num_select,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "precision": "fp16" if args.fp16 else "fp32",
        "rfdetr_version": "1.11.2",
    }
    name = Path(args.checkpoint).stem + ("_fp16" if args.fp16 else "_fp32")
    Path(args.checkpoint).with_suffix(".classes.json").write_text(json.dumps({
        key: metadata[key] for key in ["model_type", "resolution", "class_names",
                                      "class_id_offset", "background_class_id"]
    }, indent=2) + "\n")
    engine = backend.model.export(
        format="tensorrt", output_dir=args.output_dir, output_name=name,
        batch_size=1, dynamic_batch=False, fp16=args.fp16, verbose=False,
    )
    Path(engine).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.validation_images:
        metadata["validation"] = validate_export(backend, str(engine), args.validation_images, args.threshold)
        Path(engine).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(engine)


if __name__ == "__main__":
    main()
