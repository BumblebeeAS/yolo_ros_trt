from typing import Dict, List

import supervision as sv
from foxglove_msgs.msg import (
    Color,
    ImageAnnotations,
    Point2,
    PointsAnnotation,
    TextAnnotation,
)
from std_msgs.msg import Header
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.geometry.core import Position
from ultralytics.engine.results import Boxes, Keypoints, Masks, Results
from yolo_msgs.msg import (
    BoundingBox2D,
    Detection,
    DetectionArray,
    KeyPoint2D,
    KeyPoint2DArray,
    Mask,
    Point2D,
)

# Source: https://supervision.roboflow.com/draw/color/#supervision.draw.color.ColorPalette.DEFAULT
DEFAULT_COLOR_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


def get_image_annotations_from_detections(
    detections: sv.Detections,
    header: Header,
    colors: List[str] = DEFAULT_COLOR_PALETTE,
    font_size: float = 50.0,
    display_tracker_id: bool = False,
) -> ImageAnnotations:
    def hex_to_rgba(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip("#")
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
        a = 255
        return (r, g, b, a)

    image_annotations = ImageAnnotations()
    bbox_top_left_positions = detections.get_anchors_coordinates(Position.TOP_LEFT)

    if detections.mask is not None:
        for i in range(len(detections.mask)):
            class_id = detections.class_id[i]
            class_label = detections[CLASS_NAME_DATA_FIELD][i]
            confidence = detections.confidence[i]
            bbox_top_left_position = bbox_top_left_positions[i]
            mask = detections.mask[i]

            polygons = sv.detection.utils.mask_to_polygons(mask)
            for polygon in polygons:
                points = [Point2(x=float(x), y=float(y)) for x, y in polygon]
                outline_color = hex_to_rgba(colors[class_id % len(colors)])
                r, g, b, a = map(lambda x: x / 255.0, outline_color)

                points_annotation = PointsAnnotation(
                    timestamp=header.stamp,
                    type=PointsAnnotation.LINE_STRIP,
                    points=points,
                    outline_color=Color(r=r, g=g, b=b, a=a),
                    thickness=5.0,
                )
                image_annotations.points.append(points_annotation)

            text_position_x, text_position_y = bbox_top_left_position
            if display_tracker_id and detections.tracker_id is not None:
                tracker_id = detections.tracker_id[i]
                text = f"{class_label} {confidence:.2f} {tracker_id}"
            else:
                text = f"{class_label} {confidence:.2f}"

            text_annotation = TextAnnotation(
                timestamp=header.stamp,
                position=Point2(x=float(text_position_x), y=float(text_position_y)),
                text=text,
                font_size=font_size,
                text_color=Color(r=1.0, g=1.0, b=1.0, a=1.0),
                background_color=Color(r=r, g=g, b=b, a=a),
            )
            image_annotations.texts.append(text_annotation)

    return image_annotations


def parse_hypothesis(results: Results, class_names: Dict[int, str]) -> List[Dict]:
    hypothesis_list = []

    if results.boxes:
        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": class_names[int(box_data.cls)],
                "score": float(box_data.conf),
                "id": str(int(box_data.id)) if box_data.id is not None else "null",
            }
            hypothesis_list.append(hypothesis)

    elif results.obb:
        for i in range(results.obb.cls.shape[0]):
            hypothesis = {
                "class_id": int(results.obb.cls[i]),
                "class_name": class_names[int(results.obb.cls[i])],
                "score": float(results.obb.conf[i]),
                "id": (
                    str(int(results.obb.id)) if results.obb.id is not None else "null"
                ),
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
            aux_msg.id = hypothesis[i]["id"]

            aux_msg.bbox = boxes[i]

        if results.masks and masks:
            aux_msg.mask = masks[i]

        if results.keypoints and keypoints:
            aux_msg.keypoints = keypoints[i]

        detections_msg.detections.append(aux_msg)

    detections_msg.header = header
    return detections_msg
