import gc
import time
from pathlib import Path

import rclpy
import supervision as sv
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


class YoloNode(LifecycleNode):
    def __init__(self, name="yolo_node") -> None:
        super().__init__(name)

        self.declare_parameter("model_path", "")

        # YOLO predict parameters
        # See https://docs.ultralytics.com/usage/cfg/#predict-settings
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("activate_on_start", False)

        self.declare_parameter("input_image_topic", "image")
        self.declare_parameter("output_detections_topic", "yolo/detections")
        self.declare_parameter("output_annotations_topic", "yolo/annotations")
        self.declare_parameter("display_tracker_id", False)
        self.declare_parameter("font_size", 50.0)

        # Latency profiling: log a windowed mean/max breakdown of the per-frame
        # pipeline (image age, convert, inference, postprocess+publish, annotate)
        # every `profiling_log_every` frames. Cheap (perf_counter only); set
        # enable_profiling:=false to silence.
        self.declare_parameter("enable_profiling", False)
        self.declare_parameter("profiling_log_every", 30)

        # Check if the model file exists
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        my_file = Path(model_path)
        if not my_file.is_file():
            raise ValueError(f"Model file '{model_path}' does not exist")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # Create publishers
        self.bridge = CvBridge()
        output_detections_topic = (
            self.get_parameter("output_detections_topic")
            .get_parameter_value()
            .string_value
        )
        output_annotations_topic = (
            self.get_parameter("output_annotations_topic")
            .get_parameter_value()
            .string_value
        )
        self.detections_publisher = self.create_lifecycle_publisher(
            DetectionArray, output_detections_topic, qos_profile=qos_profile_sensor_data
        )
        self.debug_annotations_publisher = self.create_lifecycle_publisher(
            ImageAnnotations,
            output_annotations_topic,
            qos_profile=qos_profile_sensor_data,
        )

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Load the model
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        conf = self.get_parameter("conf").get_parameter_value().double_value
        iou = self.get_parameter("iou").get_parameter_value().double_value
        agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )

        try:
            self.model = YOLO(model_path, task="segment")
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{model_path}' does not exists")
            return TransitionCallbackReturn.ERROR
        self.get_logger().info(f"[{self.get_name()}] Model loaded: {model_path}")

        self.model_predict = lambda image: self.model.predict(
            image, conf=conf, iou=iou, agnostic_nms=agnostic_nms
        )
        self.get_logger().info(
            f"[{self.get_name()}] Model predict function created with conf={conf}, iou={iou}, agnostic_nms={agnostic_nms}"
        )
        self.class_names = self.model.names
        self.get_logger().info(
            f"[{self.get_name()}] Class names set up: {self.class_names}"
        )

        # Create subscribers
        input_image_topic = (
            self.get_parameter("input_image_topic").get_parameter_value().string_value
        )
        sub_qos_profile = qos_profile_sensor_data
        sub_qos_profile.depth = 1
        self.image_subscriber = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            sub_qos_profile,
        )
        self.get_logger().info(
            f"[{self.get_name()}] Subscribed to input image topic: {input_image_topic}"
        )

        # Profiling state
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

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.model
        del self.model_predict
        del self.class_names

        gc.collect()

        self.destroy_subscription(self.image_subscriber)
        self.image_subscriber = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self.detections_publisher)
        self.detections_publisher = None
        self.destroy_publisher(self.debug_annotations_publisher)
        self.debug_annotations_publisher = None

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    _PROF_KEYS = (
        "age_ms",
        "convert_ms",
        "infer_ms",
        "post_pub_ms",
        "annot_ms",
        "e2e_ms",
    )

    def _prof_reset(self) -> None:
        self._prof_n = 0
        self._prof_sum = {k: 0.0 for k in self._PROF_KEYS}
        self._prof_max = {k: 0.0 for k in self._PROF_KEYS}

    def _prof_record(self, **vals: float) -> None:
        self._prof_n += 1
        for k, v in vals.items():
            self._prof_sum[k] += v
            if v > self._prof_max[k]:
                self._prof_max[k] = v
        if self._prof_n < self._prof_every:
            return
        n = self._prof_n
        m = {k: self._prof_sum[k] / n for k in self._PROF_KEYS}
        x = self._prof_max
        self.get_logger().info(
            f"[prof] n={n} "
            f"age {m['age_ms']:.0f}/{x['age_ms']:.0f} | "
            f"conv {m['convert_ms']:.1f}/{x['convert_ms']:.1f} | "
            f"infer {m['infer_ms']:.1f}/{x['infer_ms']:.1f} | "
            f"post+pub {m['post_pub_ms']:.1f}/{x['post_pub_ms']:.1f} | "
            f"annot {m['annot_ms']:.1f}/{x['annot_ms']:.1f} "
            f"=> e2e(stamp->det) {m['e2e_ms']:.0f}/{x['e2e_ms']:.0f} ms (mean/max)"
        )
        self._prof_reset()

    def image_callback(self, msg: Image) -> None:
        t0 = time.perf_counter()
        # Image age when YOLO starts: camera capture stamp -> now (transport+queue wait)
        age_ms = (
            self.get_clock().now() - Time.from_msg(msg.header.stamp)
        ).nanoseconds / 1e6

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        t1 = time.perf_counter()

        # .cpu() forces a GPU sync, so t2 reflects true inference completion.
        results = self.model_predict(cv_image)[0].cpu()
        t2 = time.perf_counter()

        detections = get_detections(results, msg.header, self.class_names)
        self.detections_publisher.publish(detections)
        t3 = time.perf_counter()

        # Debug annotations
        sv_detections = sv.Detections.from_ultralytics(results)
        font_size = self.get_parameter("font_size").get_parameter_value().double_value
        display_tracker_id = (
            self.get_parameter("display_tracker_id").get_parameter_value().bool_value
        )
        image_annotations = get_image_annotations_from_detections(
            sv_detections,
            msg.header,
            font_size=font_size,
            display_tracker_id=display_tracker_id,
        )
        self.debug_annotations_publisher.publish(image_annotations)
        t4 = time.perf_counter()

        if self._profiling:
            self._prof_record(
                age_ms=age_ms,
                convert_ms=(t1 - t0) * 1e3,
                infer_ms=(t2 - t1) * 1e3,
                post_pub_ms=(t3 - t2) * 1e3,
                annot_ms=(t4 - t3) * 1e3,
                e2e_ms=age_ms + (t3 - t0) * 1e3,
            )


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
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
