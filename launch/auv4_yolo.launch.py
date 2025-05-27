import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


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

    symbol_yolo_node = Node(
        package="yolo_ros_trt",
        executable="yolo_node",
        name="symbol_yolo_node",
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

    gate_compression_node = Node(
        package="image_transport",
        executable="republish",
        name="gate_compression_node",
        arguments=["raw", "compressed"],
        output="screen",
        parameters=[{"out.jpeg_quality": 50}],
        namespace="/auv4/front_cam",
        remappings=[
            ("in", "gate/yolo/image"),
            ("out/compressed", "gate/yolo/image/compressed"),
        ],
    )

    symbol_compression_node = Node(
        package="image_transport",
        executable="republish",
        name="symbol_compression_node",
        arguments=["raw", "compressed"],
        output="screen",
        parameters=[{"out.jpeg_quality": 50}],
        namespace="/auv4/front_cam",
        remappings=[
            ("in", "symbol/yolo/image"),
            ("out/compressed", "symbol/yolo/image/compressed"),
        ],
    )

    return LaunchDescription(
        [
            gate_yolo_node,
            symbol_yolo_node,
            gate_compression_node,
            symbol_compression_node,
            gate_pose_estimator_node,
        ]
    )
