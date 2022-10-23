from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union

from loguru import logger
from easydict import EasyDict

from app.yolov7_wrapper import DetectorObject

YOLO_CLASS_NAMES = {"R_Signal": "Red", "G_Signal": "Green"}


class TrafficLight(Enum):
    Red = 0
    Green = 1
    NotFound = 2

    @staticmethod
    def fromYoloClassName(className: str):
        if className in YOLO_CLASS_NAMES:
            return TrafficLight[YOLO_CLASS_NAMES[className]]
        else:
            raise ValueError("invalid className for traffic light!")


args = EasyDict(
    {
        "track_points": "bbox_distance",
    }
)


@dataclass
class Position:
    x: float
    y: float
    heading: float
    speed: float

    def similar_to(self, to: "Position"):
        heading_diff = (self.heading - to.heading) % 360
        heading_diff = (
            360 - heading_diff if heading_diff > 180 else heading_diff
        )  # 두 방향의 짧은 쪽 각 차이
        return (
            abs(heading_diff) < 10
            and self.speed < 0.5
            # and (abs(self.x - to.x) < 0.00001 and abs(self.y - to.y) < 0.00001)
        )


class StateMachine:

    is_guiding_crossroad = False
    last_trafficlight = TrafficLight.Red
    current_trafficlight = TrafficLight.Red

    guides: List[str] = []

    @property
    def cross(self):
        return self._cross > 0

    @property
    def red(self):
        return self._red > 0

    @property
    def green(self):
        return self._green > 0

    def __init__(self, should_light_exist=True, use_gps=True):
        self.should_light_exist = should_light_exist

        self.cross_not_disappear_if_position_similar = use_gps
        self.position: Position = None
        self._cross, self._red, self._green = 0, 0, 0

    def serialize(self) -> str:
        return ""

    def newFrame(
        self,
        frame_data: List[DetectorObject],
        position: Optional[Position] = None,
    ):
        self.guides = []

        if self.position is None:
            self.position = position

        cross, red, green = self.detect_class_cases(
            frame_data, cases=["Zebra_Cross", "R_Signal", "G_Signal"]
        )

        if position is None:
            self._cross = cross
        elif position.similar_to(self.position):

            # 위치가 변하지 않으면, 횡단보도가 사라질 리 없으므로 횡단보도 박스가 없어져도 반영하지 않는다.
            if not cross and self.cross_not_disappear_if_position_similar:
                pass
            else:
                self._cross = cross

            logger.info(
                f"new position {position} similar; cross {cross}, self.cross {self.cross}"
            )
        else:
            self._cross = cross
            self.position = position
            logger.info(
                f"new position {position} *different*; cross {cross}, self.cross {self.cross}"
            )

        if red:
            self._red = 6
        else:
            self._red = max(0, self._red - 1)

        if green:
            self._green = 6
        else:
            self._green = max(0, self._green - 1)

        logger.info(f"red {self._red}, green {self._green}")

        self.process_state()

    def process_state(self):

        if self.should_light_exist:  # 신있횡

            if self.cross and (self.red or self.green):  # 횡단보도, 신호등 모두 있을 때
                self.current_trafficlight = (
                    TrafficLight.Red if self.red else TrafficLight.Green
                )

                if not self.is_guiding_crossroad:
                    self.start_guiding_crossroad()
                elif self.current_trafficlight != self.last_trafficlight:
                    self.on_trafficlight_changed()

            elif self.cross and not (self.red or self.green):  # 횡단보도만 있을 때
                self.current_trafficlight = TrafficLight.NotFound

                if not self.is_guiding_crossroad:
                    self.start_guiding_crossroad()
                elif self.current_trafficlight != self.last_trafficlight:
                    self.on_trafficlight_changed()

            elif not self.cross:  # 횡단보도 하나도 없을 때

                if self.is_guiding_crossroad:
                    self.on_end_crossroad()

            self.last_trafficlight = self.current_trafficlight

        elif not self.should_light_exist:  # 신없횡
            if self.cross and not self.is_guiding_crossroad:
                self.start_guiding_crossroad()
            elif not self.cross and self.is_guiding_crossroad:
                self.on_end_crossroad()

    def detect_class_cases(
        self,
        frame_data: List[DetectorObject],
        cases: List[Union[str, List[str]]],
    ) -> List[bool]:
        class_count_map = Counter(obj.name for obj in frame_data)

        cases_existence_list = []
        for case in cases:
            case = [case] if isinstance(case, str) else case

            case_exsists = all(
                class_count_map[class_name] for class_name in case
            )
            cases_existence_list.append(case_exsists)

        return cases_existence_list

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

    def start_guiding_crossroad(self, initial_mention=True):
        self.is_guiding_crossroad = True
        if not self.should_light_exist:
            self.guide("무신호 횡단보도 감지됨.")
            return

        self.last_trafficlight = self.current_trafficlight

        if initial_mention:
            self.guide("횡단보도 감지됨.")

        if self.current_trafficlight == TrafficLight.Red:
            self.guide("빨간불입니다.")
        elif self.current_trafficlight == TrafficLight.Green:
            self.guide("초록불입니다.")
        elif self.current_trafficlight == TrafficLight.NotFound:
            self.guide("신호등 인식 불가! 시야를 움직여 주세요.")

    def on_end_crossroad(self):
        self.is_guiding_crossroad = False
        self.guide("횡단보도가 시야에서 사라졌습니다.")

    def on_trafficlight_changed(self):

        if self.last_trafficlight == TrafficLight.NotFound:
            self.guide("신호등 정상 인식.")
            self.start_guiding_crossroad(initial_mention=False)
        else:
            if self.current_trafficlight == TrafficLight.Red:
                self.guide("신호가 빨간불로 바뀌었습니다.")
            elif self.current_trafficlight == TrafficLight.Green:
                self.guide("신호가 초록불로 바뀌었습니다.")
            elif self.current_trafficlight == TrafficLight.NotFound:
                self.guide("신호등 인식 안됨!")

    def guide(self, message: str):
        # print(message)
        self.guides.append(message)
