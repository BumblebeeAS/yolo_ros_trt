"""Multi-model YOLO lifecycle node.

Runs several YOLO models that all consume the SAME image topic from a single
subscription and a single decode, instead of one node (and one subscription +
one cv_bridge decode) per model. On a shared-memory camera pipeline this removes
the per-model fan-out cost: with N separate yolo_node processes the debayer must
publish the 9.4 MB color/image to N readers and each reader decodes it
independently, which saturates CPU/memory bandwidth and starves the camera
debayer. Collapsing the models that are always activated together into one node
cuts that to a single reader + single decode.

This keeps the existing lifecycle framework unchanged: it is still one
LifecycleNode with the standard configure/activate/deactivate transitions, so
the vision lifecycle manager and mission planner drive it exactly as before --
only the launch and the manager's node_names list change (one node instead of
several). It is generic: the set of models is given by the `models` string-array
param, with per-model settings under `<model>.*`, so it is not tied to any
particular detectors.

Params:
  input_image_topic (str)          shared input image topic
  models (str[])                   model keys to run, e.g. ["gate", "symbol"]
  font_size (float)                annotation font size (shared)
  display_tracker_id (bool)        annotation tracker id (shared)
  activate_on_start (bool)         auto-activate after configure
  enable_profiling (bool)          log per-window latency breakdown
  profiling_log_every (int)        frames per profiling window
  <model>.model_path (str)         path to the model/engine file
  <model>.conf (float)             confidence threshold
  <model>.iou (float)              NMS IoU threshold
  <model>.agnostic_nms (bool)      class-agnostic NMS
  <model>.detections_topic (str)   output DetectionArray topic
  <model>.annotations_topic (str)  output ImageAnnotations topic
"""

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
import supervision as sv
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageAnnotations
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.utils.yolo_node_helper import (
    get_detections,
    get_image_annotations_from_detections,
)


@dataclass
class _ModelCfg:
    name: str
    model_path: str
    conf: float
    iou: float
    agnostic_nms: bool
    detections_topic: str
    annotations_topic: str
    mode: str  # "predict" (stateless) or "track" (stateful, assigns track IDs)
    tracker: str  # tracker config (used only when mode == "track")


class _ModelInstance:
    """One YOLO model: config, loaded engine, and its output publishers.

    Publishers are created once at configure time (lifecycle publishers); the
    engine is loaded on activate and freed on deactivate.
    """

    def __init__(self, node: LifecycleNode, cfg: _ModelCfg) -> None:
        self._node = node
        self.cfg = cfg
        self.model = None
        self._infer = None
        # Plain publishers (not lifecycle publishers): inference only runs while
        # the node is active because the image subscription is created/destroyed
        # in on_activate/on_deactivate, so publish() is only ever called when
        # active anyway.
        self.det_pub = node.create_publisher(
            DetectionArray, cfg.detections_topic, qos_profile_sensor_data
        )
        self.ann_pub = node.create_publisher(
            ImageAnnotations, cfg.annotations_topic, qos_profile_sensor_data
        )

    def load(self) -> None:
        self.model = YOLO(self.cfg.model_path, task="segment")
        # Build the inference callable once: track() (stateful, persists track IDs
        # across frames) or predict() (stateless). Each model owns its YOLO object,
        # so per-model tracker state is independent.
        if self.cfg.mode == "track":
            tracker = self.cfg.tracker
            if not os.path.isabs(tracker):
                tracker = str(
                    Path(get_package_share_directory("yolo_ros_trt"))
                    / "config"
                    / tracker
                )
            self._infer = lambda image: self.model.track(
                image,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                agnostic_nms=self.cfg.agnostic_nms,
                persist=True,
                tracker=tracker,
            )
        else:
            self._infer = lambda image: self.model.predict(
                image,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                agnostic_nms=self.cfg.agnostic_nms,
            )

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._infer = None

    def destroy(self) -> None:
        if self.det_pub is not None:
            self._node.destroy_publisher(self.det_pub)
            self.det_pub = None
        if self.ann_pub is not None:
            self._node.destroy_publisher(self.ann_pub)
            self.ann_pub = None

    def infer_and_publish(
        self, cv_image, header, font_size, display_tracker_id
    ) -> None:
        results = self._infer(cv_image)[0].cpu()

        detections = get_detections(results, header, self.model.names)
        self.det_pub.publish(detections)

        sv_detections = sv.Detections.from_ultralytics(results)
        image_annotations = get_image_annotations_from_detections(
            sv_detections,
            header,
            font_size=font_size,
            display_tracker_id=display_tracker_id,
        )
        self.ann_pub.publish(image_annotations)


class MultiYoloNode(LifecycleNode):
    def __init__(self, name: str = "multi_yolo_node") -> None:
        super().__init__(name)

        # Shared params
        self.declare_parameter("input_image_topic", "image")
        self.declare_parameter("activate_on_start", False)
        self.declare_parameter("font_size", 50.0)
        self.declare_parameter("display_tracker_id", False)
        self.declare_parameter("enable_profiling", True)
        self.declare_parameter("profiling_log_every", 30)

        # Model list (required) + per-model params.
        self.declare_parameter("models", rclpy.Parameter.Type.STRING_ARRAY)
        model_names = list(
            self.get_parameter("models").get_parameter_value().string_array_value
        )
        if not model_names:
            raise ValueError(
                "multi_yolo_node requires a non-empty 'models' string array param"
            )

        self._cfgs: list[_ModelCfg] = []
        for n in model_names:
            self.declare_parameter(f"{n}.model_path", "")
            self.declare_parameter(f"{n}.conf", 0.25)
            self.declare_parameter(f"{n}.iou", 0.7)
            self.declare_parameter(f"{n}.agnostic_nms", False)
            self.declare_parameter(f"{n}.detections_topic", f"{n}/yolo/detections")
            self.declare_parameter(f"{n}.annotations_topic", f"{n}/yolo/annotations")
            self.declare_parameter(f"{n}.mode", "predict")  # "predict" or "track"
            self.declare_parameter(f"{n}.tracker", "bytetrack.yaml")

            model_path = (
                self.get_parameter(f"{n}.model_path").get_parameter_value().string_value
            )
            if not Path(model_path).is_file():
                raise ValueError(
                    f"Model file '{model_path}' for model '{n}' does not exist"
                )

            mode = self.get_parameter(f"{n}.mode").get_parameter_value().string_value
            if mode not in ("predict", "track"):
                raise ValueError(
                    f"model '{n}' mode must be 'predict' or 'track', got '{mode}'"
                )

            self._cfgs.append(
                _ModelCfg(
                    name=n,
                    model_path=model_path,
                    conf=self.get_parameter(f"{n}.conf")
                    .get_parameter_value()
                    .double_value,
                    iou=self.get_parameter(f"{n}.iou")
                    .get_parameter_value()
                    .double_value,
                    agnostic_nms=self.get_parameter(f"{n}.agnostic_nms")
                    .get_parameter_value()
                    .bool_value,
                    detections_topic=self.get_parameter(f"{n}.detections_topic")
                    .get_parameter_value()
                    .string_value,
                    annotations_topic=self.get_parameter(f"{n}.annotations_topic")
                    .get_parameter_value()
                    .string_value,
                    mode=mode,
                    tracker=self.get_parameter(f"{n}.tracker")
                    .get_parameter_value()
                    .string_value,
                )
            )

        self._models: list[_ModelInstance] = []
        self.image_subscriber = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"[{self.get_name()}] Configuring {len(self._cfgs)} model(s)..."
        )
        self.bridge = CvBridge()
        self._models = [_ModelInstance(self, cfg) for cfg in self._cfgs]
        self._prof_reset()
        super().on_configure(state)
        self.get_logger().info(
            f"[{self.get_name()}] Configured models: {[c.name for c in self._cfgs]}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        for m in self._models:
            m.load()
            self.get_logger().info(f"[{self.get_name()}] Loaded model '{m.cfg.name}'")

        input_image_topic = (
            self.get_parameter("input_image_topic").get_parameter_value().string_value
        )
        self._profiling = (
            self.get_parameter("enable_profiling").get_parameter_value().bool_value
        )
        self._prof_every = max(
            1,
            self.get_parameter("profiling_log_every")
            .get_parameter_value()
            .integer_value,
        )
        self._prof_reset()

        # Single shared subscription -> single decode for all models.
        self.image_subscriber = self.create_subscription(
            Image, input_image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"[{self.get_name()}] Subscribed to '{input_image_topic}' for "
            f"{len(self._models)} model(s)"
        )

        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        if self.image_subscriber is not None:
            self.destroy_subscription(self.image_subscriber)
            self.image_subscriber = None
        for m in self._models:
            m.unload()
        gc.collect()
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        for m in self._models:
            m.destroy()
        self._models = []
        super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Shutdown can be entered from ANY state (incl. active), so tear down
        # everything idempotently: drop the subscription, free every engine +
        # tracker state, then destroy publishers. unload()/destroy() guard against
        # being called when already cleaned, so this is safe regardless of which
        # transitions ran before.
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        if getattr(self, "image_subscriber", None) is not None:
            self.destroy_subscription(self.image_subscriber)
            self.image_subscriber = None
        for m in self._models:
            m.unload()
            m.destroy()
        self._models = []
        gc.collect()
        super().on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg: Image) -> None:
        t0 = time.perf_counter()
        age_ms = (
            self.get_clock().now() - Time.from_msg(msg.header.stamp)
        ).nanoseconds / 1e6

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        t_decode = time.perf_counter()

        # Sequential inference. The models share one GPU, which serializes their
        # kernels anyway, so running them in sequence costs the same wall time as
        # threads would while keeping the node simple.
        per_model_ms = {}
        for m in self._models:
            ts = time.perf_counter()
            m.infer_and_publish(
                cv_image,
                msg.header,
                self.get_parameter("font_size").get_parameter_value().double_value,
                self.get_parameter("display_tracker_id")
                .get_parameter_value()
                .bool_value,
            )
            per_model_ms[m.cfg.name] = (time.perf_counter() - ts) * 1e3
        t_end = time.perf_counter()

        if self._profiling:
            self._prof_record(
                age_ms=age_ms,
                decode_ms=(t_decode - t0) * 1e3,
                infer_ms=(t_end - t_decode) * 1e3,
                e2e_ms=age_ms + (t_end - t0) * 1e3,
                per_model_ms=per_model_ms,
            )

    # --- lightweight windowed profiling -------------------------------------

    def _prof_reset(self) -> None:
        self._prof_n = 0
        self._prof_sum = {
            "age_ms": 0.0,
            "decode_ms": 0.0,
            "infer_ms": 0.0,
            "e2e_ms": 0.0,
        }
        self._prof_max = {
            "age_ms": 0.0,
            "decode_ms": 0.0,
            "infer_ms": 0.0,
            "e2e_ms": 0.0,
        }
        self._prof_model_sum: dict[str, float] = {}

    def _prof_record(self, age_ms, decode_ms, infer_ms, e2e_ms, per_model_ms) -> None:
        self._prof_n += 1
        for k, v in (
            ("age_ms", age_ms),
            ("decode_ms", decode_ms),
            ("infer_ms", infer_ms),
            ("e2e_ms", e2e_ms),
        ):
            self._prof_sum[k] += v
            if v > self._prof_max[k]:
                self._prof_max[k] = v
        for name, v in per_model_ms.items():
            self._prof_model_sum[name] = self._prof_model_sum.get(name, 0.0) + v
        if self._prof_n < self._prof_every:
            return
        n = self._prof_n
        m = {k: self._prof_sum[k] / n for k in self._prof_sum}
        x = self._prof_max
        models_str = " ".join(
            f"{name} {self._prof_model_sum[name] / n:.1f}"
            for name in self._prof_model_sum
        )
        self.get_logger().info(
            f"[prof] n={n} age {m['age_ms']:.0f}/{x['age_ms']:.0f} | "
            f"decode {m['decode_ms']:.1f}/{x['decode_ms']:.1f} | "
            f"infer {m['infer_ms']:.1f}/{x['infer_ms']:.1f} ({models_str}) "
            f"=> e2e {m['e2e_ms']:.0f}/{x['e2e_ms']:.0f} ms (mean/max)"
        )
        self._prof_reset()


def main(args=None):
    rclpy.init(args=args)
    node = MultiYoloNode()
    node.trigger_configure()
    if node.get_parameter("activate_on_start").get_parameter_value().bool_value:
        node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
