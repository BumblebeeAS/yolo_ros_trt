from typing import Dict, List

import numpy as np
import supervision as sv
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from ultralytics.engine.results import Boxes, Keypoints, Masks, Results
from visualization_msgs.msg import ImageMarker
from yolo_msgs.msg import (
    BoundingBox2D,
    Detection,
    DetectionArray,
    KeyPoint2D,
    KeyPoint2DArray,
    Mask,
    Point2D,
)


def points_list_to_ros_points(points_list: np.ndarray) -> List[Point]:
    def create_point2d(x: float, y: float) -> Point:
        p = Point()
        p.x = x
        p.y = y
        return p

    ros_points = []
    points_list = points_list.astype(float)
    for point in points_list:
        point = create_point2d(*point)
        ros_points.append(point)

    return ros_points


def get_image_marker_msg_array(
    detections: sv.Detections, header: Header
) -> ImageMarkerArray:
    img_marker_msg_array = ImageMarkerArray()

    if detections.mask is not None:
        for class_id, mask in zip(detections.class_id, detections.mask):
            img_marker_msg = ImageMarker()
            img_marker_msg.header = header
            img_marker_msg.id = int(class_id)
            img_marker_msg.type = ImageMarker.POLYGON
            points_list = sv.detection.utils.mask_to_polygons(mask)[0]
            ros_points = points_list_to_ros_points(points_list)
            img_marker_msg.points = ros_points
            outline_color = ColorRGBA()
            outline_color.r = 0.0
            outline_color.g = 1.0
            outline_color.b = 0.0
            outline_color.a = 1.0
            img_marker_msg.outline_color = outline_color
            img_marker_msg.scale = 1.0

            img_marker_msg_array.markers.append(img_marker_msg)

    return img_marker_msg_array


def parse_hypothesis(results: Results, class_names: Dict[int, str]) -> List[Dict]:
    hypothesis_list = []

    if results.boxes:
        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": class_names[int(box_data.cls)],
                "score": float(box_data.conf),
            }
            hypothesis_list.append(hypothesis)

    elif results.obb:
        for i in range(results.obb.cls.shape[0]):
            hypothesis = {
                "class_id": int(results.obb.cls[i]),
                "class_name": class_names[int(results.obb.cls[i])],
                "score": float(results.obb.conf[i]),
            }
            hypothesis_list.append(hypothesis)

    return hypothesis_list


def parse_boxes(results: Results) -> List[BoundingBox2D]:
    boxes_list = []

    if results.boxes:
        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

    elif results.obb:
        for i in range(results.obb.cls.shape[0]):
            msg = BoundingBox2D()

            # get boxes values
            box = results.obb.xywhr[i]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.center.theta = float(box[4])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

    return boxes_list


def parse_masks(results: Results) -> List[Mask]:
    masks_list = []

    def create_point2d(x: float, y: float) -> Point2D:
        p = Point2D()
        p.x = x
        p.y = y
        return p

    mask: Masks
    for mask in results.masks:

        msg = Mask()

        msg.data = [
            create_point2d(float(ele[0]), float(ele[1])) for ele in mask.xy[0].tolist()
        ]
        msg.height = results.orig_img.shape[0]
        msg.width = results.orig_img.shape[1]

        masks_list.append(msg)

    return masks_list


def parse_keypoints(results: Results, threshold: float = 0.5) -> List[KeyPoint2DArray]:
    keypoints_list = []

    points: Keypoints
    for points in results.keypoints:

        msg_array = KeyPoint2DArray()

        if points.conf is None:
            continue

        for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

            if conf >= threshold:
                msg = KeyPoint2D()

                msg.id = kp_id + 1
                msg.point.x = float(p[0])
                msg.point.y = float(p[1])
                msg.score = float(conf)

                msg_array.data.append(msg)

        keypoints_list.append(msg_array)

    return keypoints_list


def get_detections(
    results: Results,
    header: Header,
    class_names: Dict[int, str],
    keypoints_threshold: float = 0.5,
) -> DetectionArray:
    if results.boxes or results.obb:
        hypothesis = parse_hypothesis(results, class_names)
        boxes = parse_boxes(results)

    if results.masks:
        masks = parse_masks(results)

    if results.keypoints:
        keypoints = parse_keypoints(results, keypoints_threshold)

    detections_msg = DetectionArray()

    for i in range(len(results)):

        aux_msg = Detection()

        if results.boxes or results.obb and hypothesis and boxes:
            aux_msg.class_id = hypothesis[i]["class_id"]
            aux_msg.class_name = hypothesis[i]["class_name"]
            aux_msg.score = hypothesis[i]["score"]

            aux_msg.bbox = boxes[i]

        if results.masks and masks:
            aux_msg.mask = masks[i]

        if results.keypoints and keypoints:
            aux_msg.keypoints = keypoints[i]

        detections_msg.detections.append(aux_msg)

    detections_msg.header = header
    return detections_msg
