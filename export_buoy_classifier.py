"""Export a buoy classifier .pt2 pipeline to TensorRT and check source parity."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

from yolo_ros_trt.trt_engine import TensorRTEngine


def validation_cases(shape, image_paths, samples):
    rng = np.random.default_rng(42)
    height, width, _ = shape
    frames = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read validation image: {path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    frames += [np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8)]
    frames += [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(samples)]
    boxes = [
        [width*.2, height*.1, width*.4, height*.8],
        [10.2, 20.8, 70.4, 65.1], [0, 0, width, height],
        [-10, -5, 40, 60], [width-40, height-60, width+10, height+5],
        [10, 10, 10, 20], [20, 20, 10, 10],
        [width+10, 10, width+40, 60], [float("nan"), 10, 40, 60],
    ]
    for image in frames:
        if image.shape != tuple(shape):
            raise ValueError(f"Validation images must have shape {shape}, got {image.shape}")
        for box in boxes:
            yield image, np.asarray(box, np.float32)


def validate(source, engine, cases, tolerance):
    count, valid_count, max_error = 0, 0, 0.0
    cpu_ms, trt_ms = [], []
    with torch.inference_mode():
        for image, box in cases:
            start = time.perf_counter()
            expected, expected_valid = source(torch.from_numpy(image), torch.from_numpy(box))
            cpu_time = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = engine({"image": image, "box": box})
            engine_time = (time.perf_counter() - start) * 1000
            actual, valid = result["probabilities"], bool(result["valid"].item())
            if valid != bool(expected_valid):
                raise RuntimeError(f"Validity differs for box {box.tolist()}")
            count += 1
            if not valid:
                continue
            reference = expected.numpy()
            error = float(np.abs(reference - actual).max())
            max_error = max(error, max_error)
            if not np.isfinite(actual).all() or error > tolerance:
                raise RuntimeError(f"Probability parity failed: max error {error}, tolerance {tolerance}")
            if reference.argmax() != actual.argmax():
                raise RuntimeError("Predicted class differs")
            valid_count += 1
            cpu_ms.append(cpu_time)
            trt_ms.append(engine_time)
    return {"cases": count, "valid_cases": valid_count, "max_probability_error": max_error,
            "tolerance": tolerance,
            "cpu_median_ms": float(np.median(cpu_ms)), "trt_median_ms": float(np.median(trt_ms))}


def build_engine(exported, args):
    onnx_path = args.output.with_suffix(".onnx")
    print(f"Exporting {args.model} to {onnx_path}", flush=True)
    torch.onnx.export(exported, (), str(onnx_path), dynamo=True, external_data=False,
                      input_names=["image", "box"], output_names=["probabilities", "valid"],
                      opset_version=18)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    print(f"Building {args.precision} TensorRT engine ({network.num_layers} layers)", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Torch export with embedded contract.json")
    parser.add_argument("--output", type=Path, required=True, help="Target .trt engine")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--reuse-engine", type=Path, help="Revalidate an engine without rebuilding it")
    parser.add_argument("--workspace-gib", type=float, default=1)
    parser.add_argument("--validation-images", type=Path, nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=4, help="Additional seeded random frames")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.samples < 1 or args.workspace_gib <= 0 or args.atol <= 0:
        parser.error("Invalid samples, workspace or tolerance")
    if args.output.suffix not in (".trt", ".engine"):
        parser.error("Output must end in .trt or .engine")
    torch.set_num_threads(2)
    extra = {"contract.json": ""}
    exported = torch.export.load(args.model, extra_files=extra)
    contract = json.loads(extra["contract.json"])
    shape = contract["image"]["shape"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = args.reuse_engine.read_bytes() if args.reuse_engine else build_engine(exported, args)
    temporary = args.output.with_suffix(".pending.trt")
    temporary.write_bytes(serialized)
    print("Checking CPU/TensorRT parity", flush=True)
    engine = TensorRTEngine(temporary)
    try:
        report = validate(exported.module(), engine,
                          validation_cases(shape, args.validation_images, args.samples),
                          args.atol)
    finally:
        engine.close()
    metadata = {"model_type": "buoy-color-pipeline", "contract": contract,
                "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "engine_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                "precision": args.precision, "tensorrt_version": trt.__version__,
                "validation": report}
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"engine": str(args.output), **report}, indent=2))


if __name__ == "__main__":
    main()
