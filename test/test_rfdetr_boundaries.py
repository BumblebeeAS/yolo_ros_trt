"""Protect RF-DETR class indexing and BGR/RGB boundaries."""

import numpy as np
import supervision as sv
from yolo_ros_trt.rfdetr_backend import RFDetrBackend, label_detections


def test_passes_original_size_rgb_to_native_preprocessing():
    class Model:
        def predict(self, image, threshold):
            assert image.shape == (60, 100, 3)
            assert image.dtype == np.uint8
            assert image[0, 0].tolist() == [30, 20, 10]
            assert threshold == 0.4
            return sv.Detections(
                xyxy=np.array([[1, 2, 3, 4]], dtype=np.float32),
                confidence=np.array([0.9]), class_id=np.array([0]),
            )

    backend = RFDetrBackend.__new__(RFDetrBackend)
    backend.model = Model()
    backend.class_names = ["buoy"]
    backend.class_id_offset = 0
    backend.background_class_id = None
    bgr = np.full((60, 100, 3), [10, 20, 30], dtype=np.uint8)
    result = backend.predict(bgr, threshold=0.4)
    assert result.data["class_name"].tolist() == ["buoy"]

def test_legacy_coco_ids_remap_and_background_is_removed():
    detections = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]] * 3, dtype=np.float32),
        confidence=np.array([0.9, 0.9, 0.9]), class_id=np.array([0, 1, 10]),
    )
    names = ["class_" + str(i) for i in range(10)]
    result = label_detections(detections, names, class_id_offset=1, background_class_id=0)
    assert result.class_id.tolist() == [0, 9]
    assert result.data["class_name"].tolist() == ["class_0", "class_9"]
