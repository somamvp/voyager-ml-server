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
    self.guide("traffic light & crossroad detected! start guiding")
    
    if self.currentTrafficLight == TrafficLight.Red:
      self.guide("traffic light Red! stay.")
    elif self.currentTrafficLight == TrafficLight.Green:
      self.guide("traffic light Green! but, stay.")


  def onTrafficLightChanged(self):
    if self.currentTrafficLight == TrafficLight.Red:
      self.guide("changed to Red!")
    elif self.currentTrafficLight == TrafficLight.Green:
      self.guide("changed to Green!")

  def guide(self, message: str):
    print(message)
    self.guides.append(message)