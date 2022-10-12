import os, natsort, sys
from typing import List, Optional, Union
from easydict import EasyDict
import cv2

import numpy as np
import torch
import torchvision.ops.boxes as bops

import norfair
from norfair import Detection, Paths, Tracker, Video
from norfair.distances import frobenius, iou, mean_euclidean

TRACK_METHOD_PRESETS = {
    "bbox": EasyDict({
        "distance_function": iou,
        "distance_threshold": 0.8,
    }),

    "centroid": EasyDict({
        "distance_function": frobenius,
        "distance_threshold": 100,
    }),

    "bbox_distance": EasyDict({
        "distance_function": mean_euclidean,
        "distance_threshold": 200,
    }),
}


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            print("Model file error")
            exit()
            # os.system(
            #     f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(model_path)} -O {model_path}"
            # )

        # load model
        try:
            self.model = torch.hub.load("WongKinYiu/yolov7", "custom", model_path)
        except:
            raise Exception("Failed to load model from {}".format(model_path))

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def center(points):
    return [np.mean(np.array(points), axis=0)]

def detector_detections_to_norfair_detections(
    detections, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        for box in detections:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            centroid = np.array(
                [ (xmin + xmax) / 2, (ymin + ymax) / 2 ]
            )
            scores = np.array([box["confidence"]])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=box["class"],
                )
            )
    elif track_points in ["bbox", "bbox_distance"]:
        for box in detections:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            bbox = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ]
            )
            scores = np.array(
                [ box["confidence"], box["confidence"] ]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=box["class"]
                )
            )

    return norfair_detections


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points in ["bbox", "bbox_distance"]:
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


args = EasyDict({
    "detector_path": "extended_1.pt",
    "track_points": "bbox_distance",
    "img_size": 640,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "classes": None,
    # ['Zebra_Cross', 'R_Signal', 'G_Signal', 'Braille_Block', 'person', 'dog', 'tree', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'none', 'wheelchair', 'stroller', 'kickboard', 'bollard', 'manhole', 'labacon', 'bench', 'barricade', 'pot', 'table', 'chair', 'fire_hydrant', 'movable_signage', 'bus_stop'],
    "device": "cpu",
})

def track(frame: Union[str, np.ndarray], img_count: int):
    if isinstance(frame, str):
        frame = cv2.imread(frame)

    yolo_detections = model(
        frame,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.img_size,
        classes=args.classes,
    )
    detections = yolo_detections_to_norfair_detections(
        yolo_detections, track_points=args.track_points
    )

    tracked_objects = tracker.update(detections=detections)
    print(tracked_objects)

    if args.track_points == "centroid":
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
    elif args.track_points in ["bbox", "bbox_distance"]:
        norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_boxes(frame, tracked_objects, draw_labels=True, label_size=0.5)
    
    cv2.imwrite(f"./tracked/tracked_{img_count}.jpg", frame)
    return yolo_detections, detections, tracked_objects

if __name__ == "__main__":

    model = YOLO(args.detector_path, device=args.device)
    model.model.to(args.device)

    track_presets = TRACK_METHOD_PRESETS[args.track_points]

    tracker = Tracker(
        distance_function=track_presets.distance_function,
        distance_threshold=track_presets.distance_threshold,
        hit_counter_max = 2,
        initialization_delay = 1,
        filter_factory=norfair.filter.FilterPyKalmanFilterFactory(R=0.001, Q=0.001, P=0.5)
    )

    img_list = natsort.natsorted(os.listdir("../Material3"))
    cnt=0
    for img in img_list:
        _ = track("../Material3/"+img, cnt)
        cnt+=1

