import re
import cv2
import numpy as np

from typing import List, Optional
from easydict import EasyDict
from loguru import logger

import norfair
from norfair.tracker import Detection, Tracker, TrackedObject
from norfair.distances import frobenius, iou, mean_euclidean

from app.wrapper_essential import DetectorObject

TRACK_METHOD_PRESETS = {
    "bbox": EasyDict(
        {
            "distance_function": iou,
            "distance_threshold": 0.8,
        }
    ),
    "centroid": EasyDict(
        {
            "distance_function": frobenius,
            "distance_threshold": 100,
        }
    ),
    "bbox_distance": EasyDict(
        {
            "distance_function": mean_euclidean,
            "distance_threshold": 200,
        }
    ),
}


class TrackerWrapper:
    """
    norfair.Tracker 을 감싸 YOLOv7 custom detection object를 전달하고 결과를 저장
    """

    def __init__(self, track_points="bbox_distance") -> None:
        if track_points not in TRACK_METHOD_PRESETS:
            raise ValueError("track_points not supported!")

        self.track_points = track_points
        self.preset = TRACK_METHOD_PRESETS[track_points]

        self.tracker = Tracker(
            distance_function=self.preset.distance_function,
            distance_threshold=self.preset.distance_threshold,
            hit_counter_max=2,
            initialization_delay=1,
            filter_factory=norfair.filter.FilterPyKalmanFilterFactory(
                R=0.001, Q=0.001, P=0.5
            ),
        )

    def update(
        self,
        frame_data: List[DetectorObject],
        validate_zebra_cross: Optional[int] = None,
    ) -> List[DetectorObject]:
        """트래킹을 업데이트하고, 현재 트래커에서 valid한 object만을 필터링해서 반환

        Args:
            frame_data (List[DetectorObject]): `yolov7_wrapper.Detector` 에서 출력한 현재 프레임의 객체 인식 결과
            validate_zebra_cross (Optional[int], optional): 횡단보도를 멀리서부터 잡는 현상을 방지하기 위해,
                이 인자로 주어진 Y픽셀 선보다 아래에 있는 횡단보도 오브젝트만을 반환함. Defaults to None.

        Returns:
            List[DetectorObject]: 트래커에 의해 이전 프레임의 결과와 연결해, 현재 valid한 DetectorObject만을 필터링해서 반환
        """
        self.detections = self.detector_detections_to_norfair_detections(
            frame_data
        )
        self.tracked_objects = self.tracker.update(self.detections)

        if validate_zebra_cross:
            self.mark_zebra_cross_under(
                y_limit=validate_zebra_cross,
                tracked_objects=self.tracked_objects,
            )

        logger.info(f"Tracked objects: {self}")

        tracked_detector_object = [
            obj.last_detection.data  # 저장했던 DetectorObject 다시 뽑아서 리턴
            for obj in self.tracked_objects
            if getattr(obj, "has_crossed_line", True)
        ]

        return tracked_detector_object

    def mark_zebra_cross_under(
        self, y_limit: int, tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:

        for obj in tracked_objects:
            if (obj.last_detection.data.cls == 0) and not getattr(
                obj, "has_crossed_line", False
            ):
                obj.has_crossed_line = obj.estimate[:, 1].max() > y_limit
                logger.info(f"{obj.estimate}")
                if obj.has_crossed_line:
                    logger.info(
                        f"{self.tracked_object_repr(obj)} 주어진 Y값 {y_limit} 넘음; 유효한 횡단보도로 인식"
                    )

    def save_result(self, frame: np.array, save_path: str):
        if self.track_points == "centroid":
            frame = norfair.draw_points(frame, self.detections)
            frame = norfair.draw_tracked_objects(frame, self.tracked_objects)
        elif self.track_points in ["bbox", "bbox_distance"]:
            frame = norfair.draw_boxes(frame, self.detections)
            frame = norfair.draw_tracked_boxes(
                frame, self.tracked_objects, draw_labels=True, label_size=0.5
            )

        if save_path:
            cv2.imwrite(save_path, frame)

    def detector_detections_to_norfair_detections(
        self, detections: List[DetectorObject]  # bbox, centroid, bbox_distance
    ) -> List[Detection]:
        """
        convert custom DetectorObject -> norfair Detection

        Detection.data 에 원래의 DetectorObject가 저장됨
        """
        norfair_detections: List[Detection] = []

        if self.track_points == "centroid":
            for box in detections:
                xmin, ymin, xmax, ymax = box.bbox_coordinate_diagonal()
                centroid = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                scores = np.array([box.confidence])
                norfair_detections.append(
                    Detection(
                        points=centroid, scores=scores, label=box.cls, data=box
                    )
                )
        elif self.track_points in ["bbox", "bbox_distance"]:
            for box in detections:
                xmin, ymin, xmax, ymax = box.bbox_coordinate_diagonal()
                bbox = np.array(
                    [
                        [xmin, ymin],
                        [xmax, ymax],
                    ]
                )
                scores = np.array([box.confidence, box.confidence])
                norfair_detections.append(
                    Detection(
                        points=bbox, scores=scores, label=box.cls, data=box
                    )
                )

        return norfair_detections

    def __repr__(self):
        """
        norfair.TrackedObject 의 __repr__을 파싱해, list 형식으로 표시
        """
        return f"[{', '.join(self.tracked_object_repr(obj) for obj in self.tracked_objects)}]"

    def tracked_object_repr(self, tracked_obj: TrackedObject) -> str:
        repr = tracked_obj.__repr__()
        repr = re.sub(r"\033\[\d+m", "", repr)  # bold체 등 text decoration 제거
        repr = re.sub(
            r"Object_(\d+)",
            rf"{tracked_obj.last_detection.data.name}_\1",
            repr,
        )  # "Object_1()" -> "person_1()" 클래스 이름 표시

        return repr  # 앞뒤 따옴표 제거
