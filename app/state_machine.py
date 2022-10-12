from enum import Enum, auto
from typing import List, Dict, Union
from easydict import EasyDict

import norfair
from norfair.tracker import Tracker, TrackedObject

from .tracking import TRACK_METHOD_PRESETS, detector_detections_to_norfair_detections
from .voyager_metadata import YOLO_IDX_TO_NAME

YOLO_CLASS_NAMES = {
    "R_Signal": "Red", 
    "G_Signal": "Green"
}


class TrafficLight(Enum):
  Red = 0
  Green = 1

  @staticmethod
  def fromYoloClassName(className: str):
    if className in YOLO_CLASS_NAMES:
      return TrafficLight[ YOLO_CLASS_NAMES[className] ]
    else:
      raise ValueError("invalid className for traffic light!")

args = EasyDict({
    "track_points": "bbox_distance",
})

class StateMachine:

  is_guiding_crossroad = False
  last_trafficlight = TrafficLight.Red
  current_trafficlight = TrafficLight.Red

  guides = []

  def __init__(self, use_tracking=True):
    self.use_tracking = use_tracking
    if use_tracking:
      self.configure_tracking()

  def serialize(self) -> str:
    return ""


  def configure_tracking(self):
    presets = TRACK_METHOD_PRESETS[args.track_points]

    self.tracker = Tracker(
        distance_function=presets.distance_function,
        distance_threshold=presets.distance_threshold,
        hit_counter_max = 2,
        initialization_delay = 1,
        filter_factory=norfair.filter.FilterPyKalmanFilterFactory(R=0.001, Q=0.001, P=0.5)
    )


  def newFrame(self, frame_data: List[Dict[str, str]]):
    if self.use_tracking:
      detections = detector_detections_to_norfair_detections(
        frame_data, 
        track_points=args.track_points
      )
      tracked_objects = self.tracker.update(detections=detections)

      detections_with_name_only = [ {"name": YOLO_IDX_TO_NAME[obj.last_detection.label]} for obj in tracked_objects ]
      frame_data = detections_with_name_only

    self.guides = []

    if self.detect_cross_signal(frame_data):

      if not self.is_guiding_crossroad:
        self.start_guiding_crossroad()

      if self.current_trafficlight != self.last_trafficlight:
        self.on_trafficlight_changed()
      
      self.last_trafficlight = self.current_trafficlight

    elif not self.detect_classes(frame_data, ["Zebra_Cross"]):    # 횡단보도 하나도 없을 때

      if self.is_guiding_crossroad:
        self.on_end_crossroad()


  def detect_classes(self, frame_data: List[Dict[str, str]], classes: List[str]):
    class_existence_map = { class_name: False for class_name in classes }
    
    for detected_object in frame_data:
      name = detected_object["name"]
      if name in class_existence_map:
        class_existence_map[name] = True
    
    is_all_exists = all( class_existence_map.values() )
    return is_all_exists


  def detect_cross_signal(self, frame_data: List[Dict[str, str]]):
    cross_exists = False
    signal_exists = False

    for detectedObject in frame_data:
      name = detectedObject["name"]
      if name == "Zebra_Cross":
        cross_exists = True
      if name in YOLO_CLASS_NAMES:
        signal_exists = True
        self.current_trafficlight = TrafficLight.fromYoloClassName(name)

    isDetected = cross_exists and signal_exists
    return isDetected


  def start_guiding_crossroad(self):
    self.is_guiding_crossroad = True
    self.last_trafficlight = self.current_trafficlight
    self.guide("횡단보도 안내를 시작합니다.")
    
    if self.current_trafficlight == TrafficLight.Red:
      self.guide("빨간불입니다. 정지하세요.")
    elif self.current_trafficlight == TrafficLight.Green:
      self.guide("초록불입니다. 다음 신호를 기다리세요.")
    
  def on_end_crossroad(self):
    self.is_guiding_crossroad = False
    self.guide("횡단보도가 시야에서 사라졌습니다.")


  def on_trafficlight_changed(self):
    if self.current_trafficlight == TrafficLight.Red:
      self.guide("신호가 빨간불로 바뀌었습니다.")
    elif self.current_trafficlight == TrafficLight.Green:
      self.guide("신호가 초록불로 바뀌었습니다.")


  def guide(self, message: str):
    # print(message)
    self.guides.append(message)