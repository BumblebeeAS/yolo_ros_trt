import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("yolo_ros_trt"), "config", "auv4_orin.yaml"
    )
    gate_yolo_node = Node(
        package="yolo_ros_trt",
        executable="yolo_node",
        name="gate_yolo_node",
        namespace="/auv4/front_cam",
        parameters=[config],
    )

    gate_pose_estimator_node = Node(
        package="pose_estimator",
        executable="gate_pose_estimator_node",
        name="gate_pose_estimator_node",
        namespace="/auv4/front_cam",
        parameters=[config],
    )

    return LaunchDescription([gate_yolo_node, gate_pose_estimator_node])
