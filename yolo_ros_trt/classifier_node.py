"""Classify selected detection crops using class names from the exported model."""

from copy import deepcopy

import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageAnnotations
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.classifier import ExportedClassifier
from yolo_ros_trt.utils.yolo_node_helper import get_image_annotations_from_detection_array


class ClassifierNode(Node):
    def __init__(self):
        super().__init__("classifier_node")
        param = lambda name, default: self.declare_parameter(name, default).value
        model_path = param("model_path", "")
        self.classifier = ExportedClassifier(model_path, param("device", "cpu"))
        self.input_classes = param("input_classes", ["*"])
        self.label_prefix = param("label_prefix", "")
        labels = list(self.classifier.classes)
        if "unknown" not in labels:
            labels.append("unknown")  # Fallback for invalid input; preserve model IDs.
        self.class_ids = {name: index for index, name in enumerate(labels)}
        self.detection_pub = self.create_publisher(
            DetectionArray, param("output_detections_topic", "classifier/detections"), qos_profile_sensor_data
        )
        self.classification_pub = self.create_publisher(
            DetectionArray, param("output_classifications_topic", "classifier/classifications"), qos_profile_sensor_data
        )
        self.annotation_pub = self.create_publisher(
            ImageAnnotations, param("output_annotations_topic", "classifier/annotations"), qos_profile_sensor_data
        )
        self.bridge = CvBridge()
        self.image_sub = Subscriber(self, Image, param("input_image_topic", "image"), qos_profile=qos_profile_sensor_data)
        self.detection_sub = Subscriber(self, DetectionArray, param("input_detections_topic", "detections"), qos_profile=qos_profile_sensor_data)
        self.sync = TimeSynchronizer([self.image_sub, self.detection_sub], queue_size=60)
        self.sync.registerCallback(self.detections_callback)
        self.get_logger().info(f"Classifier loaded: {model_path}")

    def detections_callback(self, image_msg, detections):
        if image_msg.header.frame_id != detections.header.frame_id:
            return
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        classified = deepcopy(detections)
        classifications = DetectionArray(header=detections.header)
        for detection in classified.detections:
            if "*" not in self.input_classes and detection.class_name not in self.input_classes:
                continue
            bbox = detection.bbox
            box = [bbox.center.position.x, bbox.center.position.y, bbox.size.x, bbox.size.y]
            try:
                predicted, confidence = self.classifier.predict(image, box)
            except ValueError as error:
                self.get_logger().warning(str(error), throttle_duration_sec=5.0)
                predicted, confidence = "unknown", 0.0
            label = predicted if (np.isfinite(confidence) and predicted in self.class_ids) else "unknown"
            colored = deepcopy(detection)
            colored.class_name = label
            colored.class_id = self.class_ids[label]
            colored.score = confidence
            classifications.detections.append(colored)
            detection.class_name = f"{self.label_prefix}{label}"
        self.detection_pub.publish(classified)
        self.classification_pub.publish(classifications)
        self.annotation_pub.publish(get_image_annotations_from_detection_array(classified))


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
