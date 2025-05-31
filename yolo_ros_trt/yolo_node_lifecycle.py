import os

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
import supervision as sv
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolo_msgs.msg import DetectionArray

from yolo_ros_trt.utils.yolo_node_helper import get_detections

"""
See https://design.ros2.org/articles/node_lifecycle.html for more information on the lifecycle node and possible transitions.

Make service calls to f'{managed_node}/change_state' for make state transitions, where `managed_node` is the name of the node.
request = ChangeState.Request()
request.transition.id = transition_id
where possible `transition_id`s can be seen from https://docs.ros.org/en/humble/p/lifecycle_msgs/msg/Transition.html
from lifecycle_msgs.msg import Transition
Transition.TRANSITION_CONFIGURE

Tree Example:
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

STATE_SERVICE_NAME = "/auv4/front_cam/gate_yolo_node/change_state"
srv_activate_node = py_trees_ros.service_clients.FromConstant(
        name="Activate yolo gate node",
        service_name=STATE_SERVICE_NAME,
        service_type=ChangeState,
        service_request=ChangeState.Request(
            transition=Transition(id=Transition.TRANSITION_ACTIVATE)
        ),
        key_response=fk("activated"),
    )
Note that the `transition` field is a Transition message, which has an `id` field that can be set to the desired transition ID (e.g., `Transition.TRANSITION_ACTIVATE` for activation).
And the response value is a boolean indicating whether the transition was successful or not.

This particular node is configured on startup and thus just needs to be activated. To bring it down, deactivate it then shut it down.
"""

class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("model_name", "yolov11s_segment.engine")

        # YOLO predict parameters
        # See https://docs.ultralytics.com/usage/cfg/#predict-settings
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("agnostic_nms", False)

        self.declare_parameter("input_image_topic", "image")
        self.declare_parameter("output_detections_topic", "yolo/detections")
        self.declare_parameter("output_image_topic", "yolo/image")

    def image_callback(self, msg: Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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

        output_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        output_msg.header = msg.header

        self.image_publisher.publish(output_msg)
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring {self.get_name()}")
        # Load the model
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )

        # Annotation tools
        self.polygon_annotator = sv.PolygonAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=1.0)

        # Get parameters
        self.input_compressed_image_topic = (
            self.get_parameter("input_image_topic").get_parameter_value().string_value
        )
        self.output_detections_topic = (
            self.get_parameter("output_detections_topic")
            .get_parameter_value()
            .string_value
        )
        self.output_compressed_image_topic = (
            self.get_parameter("output_image_topic").get_parameter_value().string_value
        )

        super().on_configure(state)
        self.get_logger().info(f"{self.get_name()} configured successfully")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        model_path = os.path.join(
            get_package_share_directory("yolo_ros_trt"),
            "models",
            self.model_name,
        )
        self.model = YOLO(model_path, task="segment")
        self.model_predict = lambda image: self.model.predict(
            image, conf=self.conf, iou=self.iou, agnostic_nms=self.agnostic_nms
        )
        self.class_names = self.model.names

        # Activate/create subscribers and publishers
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, self.input_compressed_image_topic, self.image_callback, 1
        )
        self.detections_publisher = self.create_publisher(
            DetectionArray, self.output_detections_topic, 1
        )
        self.image_publisher = self.create_publisher(
            Image, self.output_compressed_image_topic, 1
        )

        super().on_activate(state)
        self.get_logger().info(f"{self.get_name()} activated successfully")
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        # Destroy subscribers and publishers
        self.destroy_subscription(self.image_subscriber)
        self.destroy_publisher(self.detections_publisher)
        self.destroy_publisher(self.image_publisher)
        self.image_subscriber = None
        self.detections_publisher = None
        self.image_publisher = None
        self.bridge = None

        # Delete the model
        del self.model

        super().on_deactivate(state)
        self.get_logger().info(f"{self.get_name()} deactivated successfully")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")
        super().on_cleanup(state)
        self.get_logger().info(f"{self.get_name()} cleaned up successfully")

        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Shutting down {self.get_name()}")
        super().on_shutdown(state)
        self.get_logger().info(f"{self.get_name()} shut down successfully")

        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    # Configure the node on startup
    node.trigger_configure()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
