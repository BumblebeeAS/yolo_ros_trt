import os

import rclpy
import supervision as sv
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.utils.yolo_node_helper import get_detections


class YoloNode(Node):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("model_name", "yolov11s_segment.engine")

        # YOLO predict parameters
        # See https://docs.ultralytics.com/usage/cfg/#predict-settings
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("agnostic_nms", False)

        self.declare_parameter("input_compressed_image_topic", "image/compressed")
        self.declare_parameter("output_detections_topic", "yolo/detections")
        self.declare_parameter("output_compressed_image_topic", "yolo/image/compressed")

        # Load the model
        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        conf = self.get_parameter("conf").get_parameter_value().double_value
        iou = self.get_parameter("iou").get_parameter_value().double_value
        agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        model_path = os.path.join(
            get_package_share_directory("yolo_ros_trt"),
            "models",
            model_name,
        )
        model = YOLO(model_path, task="segment")
        self.model_predict = lambda image: model.predict(
            image, conf=conf, iou=iou, agnostic_nms=agnostic_nms
        )
        self.class_names = model.names

        # Annotation tools
        self.polygon_annotator = sv.PolygonAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=1.0)

        # Create subscribers and publishers
        self.bridge = CvBridge()
        input_compressed_image_topic = (
            self.get_parameter("input_compressed_image_topic")
            .get_parameter_value()
            .string_value
        )
        output_detections_topic = (
            self.get_parameter("output_detections_topic")
            .get_parameter_value()
            .string_value
        )
        output_compressed_image_topic = (
            self.get_parameter("output_compressed_image_topic")
            .get_parameter_value()
            .string_value
        )
        self.image_subscriber = self.create_subscription(
            CompressedImage, input_compressed_image_topic, self.image_callback, 1
        )
        self.detections_publisher = self.create_publisher(
            DetectionArray, output_detections_topic, 1
        )
        self.image_publisher = self.create_publisher(
            CompressedImage, output_compressed_image_topic, 1
        )

    def image_callback(self, msg: CompressedImage) -> None:
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model_predict(cv_image)[0].cpu()

        detections = get_detections(results, msg.header, self.class_names)
        self.detections_publisher.publish(detections)

        # Debug image
        detections = sv.Detections.from_ultralytics(results)

        annotated_image = self.polygon_annotator.annotate(
            scene=cv_image, detections=detections
        )
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        output_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_image)
        output_msg.header = msg.header

        self.image_publisher.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
