"""Multi-model YOLO lifecycle node.

Runs several YOLO models that all consume the SAME image topic from a single
subscription and a single decode, instead of one node (and one subscription +
one cv_bridge decode) per model. On a shared-memory camera pipeline this removes
the per-model fan-out cost: with N separate yolo_node processes the debayer must
publish the 9.4 MB color/image to N readers and each reader decodes it
independently, which saturates CPU/memory bandwidth and starves the camera
debayer. Collapsing the models that are always activated together into one node
cuts that to a single reader + single decode.

Each model runs on its own worker thread fed through a single-slot "latest
frame" handoff: the subscription callback only decodes and hands the frame to
every model's slot, and each worker infers at its own pace, always on the
newest frame (older undelivered frames are overwritten, never queued). This
decouples the models' publish rates -- a cheap predict-mode model is not
throttled to the rate of an expensive track-mode one, which sequential
in-callback inference forced. The GPU serializes kernels across threads, but
inference is only part of each model's wall time (tracker + postprocess are
CPU), so the overlap is real.

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
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
import supervision as sv
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageAnnotations
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.utils.yolo_node_helper import (
    get_detections,
    get_image_annotations_from_detections,
)

# Best-effort, newest-sample-only QoS for the image subscription. Depth 1 means
# a busy node never drains a backlog of stale frames (age stays ~1 frame).
QOS_IMAGE_SUB = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
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


class _LatestFrameSlot:
    """Single-slot handoff between the subscription callback and one worker.

    put() overwrites whatever is waiting, so a slow consumer always sees the
    newest frame instead of a growing backlog. take() blocks until a frame
    arrives or the slot is closed (returns None to tell the worker to exit).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item = None
        self._closed = False

    def put(self, item) -> None:
        with self._cond:
            self._item = item
            self._cond.notify()

    def take(self):
        with self._cond:
            while self._item is None and not self._closed:
                self._cond.wait()
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()


class _WindowStats:
    """Windowed mean/max accumulator plus the window's achieved rate."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self._keys = keys
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._t_start = time.perf_counter()
        self._sum = {k: 0.0 for k in self._keys}
        self._max = {k: 0.0 for k in self._keys}

    def add(self, **vals: float) -> None:
        self.n += 1
        for k, v in vals.items():
            self._sum[k] += v
            if v > self._max[k]:
                self._max[k] = v

    def hz(self) -> float:
        dt = time.perf_counter() - self._t_start
        return self.n / dt if dt > 0.0 else 0.0

    def fmt(self, key: str) -> str:
        return f"{self._sum[key] / self.n:.1f}/{self._max[key]:.1f}"


class _ModelInstance:
    """One YOLO model: config, loaded engine, publishers, and worker thread.

    Publishers are created once at configure time; the engine is loaded on
    activate and freed on deactivate. Between start() and stop() a dedicated
    worker thread consumes frames from this model's slot, so every call into
    the YOLO object (and its tracker state) happens on that one thread.
    """

    def __init__(self, node: LifecycleNode, cfg: _ModelCfg) -> None:
        self._node = node
        self.cfg = cfg
        self.model = None
        self._infer = None
        self._slot = None
        self._thread = None
        # Plain publishers (not lifecycle publishers): the workers only run
        # between on_activate and on_deactivate, so publish() is only ever
        # called when active anyway.
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

    def start(self, profiling: bool, prof_every: int) -> None:
        self._profiling = profiling
        self._prof_every = prof_every
        self._stats = _WindowStats(("infer_ms", "e2e_ms"))
        self._slot = _LatestFrameSlot()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"yolo-{self.cfg.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._slot is not None:
            self._slot.close()
        if self._thread is not None:
            self._thread.join()
        self._slot = None
        self._thread = None

    def submit(self, cv_image, header) -> None:
        self._slot.put((cv_image, header))

    def destroy(self) -> None:
        if self.det_pub is not None:
            self._node.destroy_publisher(self.det_pub)
            self.det_pub = None
        if self.ann_pub is not None:
            self._node.destroy_publisher(self.ann_pub)
            self.ann_pub = None

    def _run_loop(self) -> None:
        while True:
            item = self._slot.take()
            if item is None:
                return
            cv_image, header = item
            t0 = time.perf_counter()
            try:
                self._infer_and_publish(cv_image, header)
            except Exception as e:
                # A bad frame must not kill the worker: log and move on.
                self._node.get_logger().error(
                    f"[{self.cfg.name}] inference failed: {e}"
                )
                continue
            if self._profiling:
                infer_ms = (time.perf_counter() - t0) * 1e3
                e2e_ms = (
                    self._node.get_clock().now() - Time.from_msg(header.stamp)
                ).nanoseconds / 1e6
                self._stats.add(infer_ms=infer_ms, e2e_ms=e2e_ms)
                if self._stats.n >= self._prof_every:
                    self._node.get_logger().info(
                        f"[prof:{self.cfg.name}] n={self._stats.n} "
                        f"infer {self._stats.fmt('infer_ms')} | "
                        f"e2e(stamp->pub) {self._stats.fmt('e2e_ms')} ms (mean/max) | "
                        f"{self._stats.hz():.1f} Hz"
                    )
                    self._stats.reset()

    def _infer_and_publish(self, cv_image, header) -> None:
        results = self._infer(cv_image)[0].cpu()

        detections = get_detections(results, header, self.model.names)
        self.det_pub.publish(detections)

        sv_detections = sv.Detections.from_ultralytics(results)
        image_annotations = get_image_annotations_from_detections(
            sv_detections,
            header,
            font_size=self._node.get_parameter("font_size")
            .get_parameter_value()
            .double_value,
            display_tracker_id=self._node.get_parameter("display_tracker_id")
            .get_parameter_value()
            .bool_value,
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
        self.declare_parameter("enable_profiling", False)
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
        super().on_configure(state)
        self.get_logger().info(
            f"[{self.get_name()}] Configured models: {[c.name for c in self._cfgs]}"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")
        profiling = (
            self.get_parameter("enable_profiling").get_parameter_value().bool_value
        )
        prof_every = max(
            1,
            self.get_parameter("profiling_log_every")
            .get_parameter_value()
            .integer_value,
        )
        self._profiling = profiling
        self._cam_stats = _WindowStats(("age_ms", "decode_ms"))
        self._prof_every = prof_every

        for m in self._models:
            m.load()
            m.start(profiling, prof_every)
            self.get_logger().info(f"[{self.get_name()}] Loaded model '{m.cfg.name}'")

        input_image_topic = (
            self.get_parameter("input_image_topic").get_parameter_value().string_value
        )
        # Single shared subscription -> single decode, fanned out to the
        # per-model worker threads via their latest-frame slots.
        self.image_subscriber = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            qos_profile=QOS_IMAGE_SUB,
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
        # Workers must exit before their engines are freed.
        for m in self._models:
            m.stop()
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
        # everything idempotently: drop the subscription, stop every worker,
        # free every engine + tracker state, then destroy publishers. stop()/
        # unload()/destroy() guard against being called when already cleaned,
        # so this is safe regardless of which transitions ran before.
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        if getattr(self, "image_subscriber", None) is not None:
            self.destroy_subscription(self.image_subscriber)
            self.image_subscriber = None
        for m in self._models:
            m.stop()
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
        decode_ms = (time.perf_counter() - t0) * 1e3

        # Fan the decoded frame out to every model's slot. Workers read the
        # array without copying (ultralytics letterboxes into its own buffer),
        # so sharing one numpy image across threads is safe.
        for m in self._models:
            m.submit(cv_image, msg.header)

        if self._profiling:
            self._cam_stats.add(age_ms=age_ms, decode_ms=decode_ms)
            if self._cam_stats.n >= self._prof_every:
                self.get_logger().info(
                    f"[prof:cam] n={self._cam_stats.n} "
                    f"age {self._cam_stats.fmt('age_ms')} | "
                    f"decode {self._cam_stats.fmt('decode_ms')} ms (mean/max) | "
                    f"{self._cam_stats.hz():.1f} Hz"
                )
                self._cam_stats.reset()


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
