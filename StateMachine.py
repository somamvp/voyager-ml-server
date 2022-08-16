from enum import Enum, auto
from typing import List, Dict

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



class StateMachine:

  isGuidingCrossroad = False
  lastTrafficLight = TrafficLight.Red
  currentTrafficLight = TrafficLight.Red

  guides = []

  def newFrame(self, frameData: List[Dict[str, str]]):
    self.guides = []

    if self.detectCrossAndSignal(frameData):

      if not self.isGuidingCrossroad:
        self.startGuidingCrossroad()

      if self.currentTrafficLight != self.lastTrafficLight:
        self.onTrafficLightChanged()
      
      self.lastTrafficLight = self.currentTrafficLight
    


  def detectCrossAndSignal(self, frameData: List[Dict[str, str]]):
    cross_exists = False
    signal_exists = False

    for detectedObject in frameData:
      name = detectedObject["name"]
      if name == "Zebra_Cross":
        cross_exists = True
      if name in YOLO_CLASS_NAMES:
        signal_exists = True
        self.currentTrafficLight = TrafficLight.fromYoloClassName(name)

    isDetected = cross_exists and signal_exists
    return isDetected


  def startGuidingCrossroad(self):
    self.isGuidingCrossroad = True
    self.lastTrafficLight = self.currentTrafficLight
    self.guide("횡단보도 안내를 시작합니다.")
    
    if self.currentTrafficLight == TrafficLight.Red:
      self.guide("빨간불입니다. 정지하세요.")
    elif self.currentTrafficLight == TrafficLight.Green:
      self.guide("초록불입니다. 다음 신호를 기다리세요.")


  def onTrafficLightChanged(self):
    if self.currentTrafficLight == TrafficLight.Red:
      self.guide("신호가 빨간불로 바뀌었습니다.")
    elif self.currentTrafficLight == TrafficLight.Green:
      self.guide("신호가 초록불로 바뀌었습니다.")

  def guide(self, message: str):
    print(message)
    self.guides.append(message)