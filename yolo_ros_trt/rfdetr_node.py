"""RF-DETR segmentation with the existing YOLO ROS message interfaces."""

import cv2
import rclpy
import supervision as sv
from yolo_msgs.msg import Detection, DetectionArray, Point2D

from yolo_ros_trt.rfdetr_backend import load_backend
from yolo_ros_trt.yolo_node import YoloNode


def to_detection_array(detections, header, image_shape):
    """Convert original-image pixel coordinates and masks to yolo_msgs."""
    output = DetectionArray()
    output.header = header
    height, width = image_shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        detection = Detection()
        detection.class_id = int(detections.class_id[i])
        detection.class_name = str(detections.data["class_name"][i])
        detection.score = float(detections.confidence[i])
        detection.id = (
            str(int(detections.tracker_id[i]))
            if detections.tracker_id is not None else "null"
        )
        detection.bbox.center.position.x = float((x1 + x2) / 2)
        detection.bbox.center.position.y = float((y1 + y2) / 2)
        detection.bbox.size.x = float(x2 - x1)
        detection.bbox.size.y = float(y2 - y1)
        if detections.mask is not None:
            mask = detections.mask[i]
            if mask.shape != (height, width):
                raise ValueError("Mask dimensions must match the original camera image")
            contours, _ = cv2.findContours(
                mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            detection.mask.height = height
            detection.mask.width = width
            # yolo_msgs/Mask represents one polygon. Keep the largest connected
            # component rather than joining disconnected outlines artificially.
            if contours:
                contour = max(contours, key=cv2.contourArea)
                detection.mask.data = [
                    Point2D(x=float(x), y=float(y)) for x, y in contour.reshape(-1, 2)
                ]
        output.detections.append(detection)
    return output


class RFDetrNode(YoloNode):
    """Reuse lifecycle, QoS and publishers with RF-DETR backends."""

    def __init__(self):
        super().__init__("rfdetr_node", "rfdetr/detections", "rfdetr/annotations")
        self.declare_parameter("tracking", False)

    def load_model(self, model_path, conf, iou, agnostic_nms):
        self.model = load_backend(model_path)
        self.class_names = self.model.class_names
        self.model_predict = lambda image: self.model.predict(image, threshold=conf)
        self.tracker = sv.ByteTrack() if self.get_parameter("tracking").value else None

    def predict_detections(self, cv_image, header):
        detections = self.model_predict(cv_image)
        if self.tracker is not None:
            detections = self.tracker.update_with_detections(detections)
        return to_detection_array(detections, header, cv_image.shape), detections


def main(args=None):
    rclpy.init(args=args)
    node = RFDetrNode()
    try:
        node.trigger_configure()
        if node.get_parameter("activate_on_start").value:
            node.trigger_activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
