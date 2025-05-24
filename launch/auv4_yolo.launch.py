import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("yolo_ros_trt"), "config", "auv4_orin.yaml"
    )
    yolo_node = Node(
        package="yolo_ros_trt",
        executable="yolo_node",
        name="yolo_node",
        namespace="/auv4/front_cam/color",
        parameters=[config],
    )

    return LaunchDescription([yolo_node])
