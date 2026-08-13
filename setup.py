import os
from glob import glob

from setuptools import setup

package_name = "yolo_ros_trt"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "src"),
            glob(os.path.join("src", "*.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="todo",
    maintainer_email="todo@todo.com",
    description="YOLO for ROS 2 with TensorRT",
    license="todo",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tracking_node = yolo_ros_trt.tracking_node:main",
            "yolo_node = yolo_ros_trt.yolo_node:main",
        ],
    },
)
