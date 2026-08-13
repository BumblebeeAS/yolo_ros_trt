from pathlib import Path

import rclpy
from ament_index_python import get_package_share_directory
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from yolo_ros_trt.yolo_node import YoloNode


class TrackingNode(YoloNode):

    def __init__(self) -> None:
        super().__init__("tracking_node")

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        super().on_activate(state)

        # Override parent's predict method
        conf = self.get_parameter("conf").get_parameter_value().double_value
        iou = self.get_parameter("iou").get_parameter_value().double_value
        agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        tracker_file_path = (
            Path(get_package_share_directory("yolo_ros_trt"))
            / "config"
            / "bytetrack.yaml"
        )
        self.model_predict = lambda image: self.model.track(
            image,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            persist=True,
            tracker=tracker_file_path,
        )

        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
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
