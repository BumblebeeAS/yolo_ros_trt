import gc

import rclpy
import supervision as sv
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageAnnotations
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.utils.yolo_node_helper import (
    get_detections,
    get_image_annotations_from_detections,
)


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("model_path", "")

        # YOLO predict parameters
        # See https://docs.ultralytics.com/usage/cfg/#predict-settings
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("agnostic_nms", False)

        self.declare_parameter("input_image_topic", "image")
        self.declare_parameter("output_detections_topic", "yolo/detections")
        self.declare_parameter("output_annotations_topic", "yolo/annotations")

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
        self.class_names = self.model.names

        # Create subscribers
        input_image_topic = (
            self.get_parameter("input_image_topic").get_parameter_value().string_value
        )
        self.image_subscriber = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

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

    def image_callback(self, msg: Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model_predict(cv_image)[0].cpu()

        detections = get_detections(results, msg.header, self.class_names)
        self.detections_publisher.publish(detections)

        # Debug annotations
        sv_detections = sv.Detections.from_ultralytics(results)
        image_annotations = get_image_annotations_from_detections(
            sv_detections, msg.header
        )
        self.debug_annotations_publisher.publish(image_annotations)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    node.trigger_configure()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
