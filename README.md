# YOLO ROS2 TensorRT

## Quickstart

### Installation

1. Build `yolo_msgs` in [bb_msgs](https://github.com/BumblebeeAS/bb_msgs/).

2. Install the required Python packages.

```bash
pip install -r requirements.txt
```

3. Note that `torch`, `torchvision` and `onnxruntime-gpu` are excluded from `requirements.txt` because the default versions are CPU-only. `ultralytics` is excluded to not override existing `torch` dependencies. Install them separately here.

For Nvidia Jetpack 6, see the [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/#run-on-jetpack-61). We have implemented this [in a Dockerfile](https://github.com/BumblebeeAS/ros2-docker/blob/master/dockerfiles/isaac_ros_jp6/Dockerfile.ultralytics_cuda).

### Run

Run `export.py` to export trained `.pt` models to `.engine` files. For example, 

```bash
python export.py yolov11s_gate_20250520_0.pt
```

Then run `yolo_node`. For example:

```bash
ros2 run yolo_ros_trt yolo_node --ros-args -p model_name:="yolov11s_gate_20250520_0.engine"
```

## Note

For object detection using YOLOv8, see [our fork of Isaac ROS Object Detection](https://github.com/BumblebeeAS/isaac_ros_object_detection) to utilize Nvidia's claim of higher efficiency due to zero copy.

## Reference

https://github.com/mgonzs13/yolo_ros
