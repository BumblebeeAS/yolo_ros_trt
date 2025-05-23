import os

import rclpy
import supervision as sv
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from src.yolo_node_helper import get_detections


class YoloNode(Node):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("model_name", "yolov11s_gate_20250520_0.engine")
        model_name = self.get_parameter("model_name").get_parameter_value().string_value

        weights_path = os.path.join(
            get_package_share_directory("yolo_ros_trt"),
            "models",
            model_name,
        )
        self.model = YOLO(weights_path, task="segment")
        self.bridge = CvBridge()
        self.polygon_annotator = sv.PolygonAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=1.0)

        self.image_subscriber = self.create_subscription(
            CompressedImage,
            "/auv4/front_cam/color/image/compressed",
            self.image_callback,
            1,
        )
        self.detections_publisher = self.create_publisher(
            DetectionArray, "/auv4/front_cam/color/image/yolo/detections", 1
        )
        self.image_publisher = self.create_publisher(
            CompressedImage, "/auv4/front_cam/color/image/yolo/image/compressed", 1
        )

    def image_callback(self, msg: CompressedImage) -> None:
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.predict(cv_image)[0].cpu()

        detections = get_detections(results, msg.header, self.model.names)
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
