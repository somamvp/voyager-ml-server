# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}]
from enum import Enum, IntEnum
import numpy as np
import math
from typing import List
from loguru import logger
from app.voyager_metadata import YOLO_NAME_TO_KOREAN, YOLO_THRES, YOLO_OBS_TYPE
from app.wrapper_essential import DetectorObject

ALPHA_TO_RANGE = 20.0 / 255.0
DEV_YOLO_BASED_WARNING = False


class ReadPriority(Enum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1
    NEAR_TO_FAR = 2


class ReadObstacle(IntEnum):
    # 안내를 원하는 객체타입을 이진마스킹 방식으로 지정 (chmod 777 하는거랑 같은 방식)
    # 0 이면 안내하는 객체 없음, 31이면 모든 객체 안내
    MOVING = 0b00001
    STATIC = 0b00010
    BRAILLE = 0b00100
    BUS_STOP = 0b01000
    OTHERS = 0b10000


class Direction(IntEnum):
    NONE = 0
    RIGHT = 1
    CENTER = 2
    LEFT = 4


obstacle_on_crosswalk = ["car", "bus", "truck", "motorcycle", "bollard"]

# Scene 문제점 : 너무 많은 객체 읽게 되는 경우
# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7


class ClockCycleStateActivator:
    # 한 세션 안에서 환경설정은 일정하다고 가정
    def __init__(self, basetime: float, settings, img_size=[640, 480]):
        self.scene_target_number: int = settings["scene_type"]
        self.target_obs: List[str] = update_target_obs(self.scene_target_number)
        self.priority: int = settings["mode"]
        self.scene_range: float = settings["scene_range"]
        self.warning_range: float = settings["warning_range"]
        self.braille_period: float = settings["braiile_period"]
        self.scene_period: float = settings["scene_period"]
        self.img_size = img_size

        self.yolo = None
        self.depth_map = None
        self.time = basetime

        # 직전 안내 시간
        self.last_scene: float = basetime
        self.last_braille: float = basetime

        # 직전 안내 State
        self.prev_braille: int = 0
        self.prev_cross: int = 0
        self.curr_braille: int = 0
        self.curr_cross: int = 0

    def timer_reset(self, time: float):
        self.time = time
        self.last_scene = time
        self.last_braille = time

    def direct(self, xc) -> int:
        if xc > self.img_size[1] * 0.25 and xc < self.img_size[1] * 0.75:
            return Direction.CENTER
        elif xc < self.img_size[1] * 0.25:
            return Direction.LEFT
        else:
            return Direction.RIGHT

    def update(self, time: float, yolo: List[DetectorObject], depth_map=None):
        self.yolo = yolo
        self.depth_map = depth_map
        self.time = time

        self.prev_braille = self.curr_braille
        self.prev_cross = self.curr_cross

        direction_list = []
        for el in self.yolo:
            if el["name"] == "Braille_Block":
                direction_list.append(self.direct(el["xc"]))
        direction_set = set(direction_list)
        self.curr_braille = 0
        for num in direction_set:
            self.curr_braille += num

        sorted_yolo = sorted(self.yolo, key=lambda x: x["ymax"], reverse=True)
        cross = None
        for el in sorted_yolo:
            if el["name"] == "Zebra_Cross":
                cross = el
                break
        self.curr_cross = (
            Direction.NONE if cross is None else self.direct(cross["xc"])
        )

    def inform_near(self) -> str:
        if self.depth_map is None:
            return ""
        else:
            pass

    def inform_on_cross(self) -> str:
        pass

    def inform_waiting_light(self) -> str:
        pass

    def inform_regular(self) -> str:

        if () or (self.time - self.last_braille > self.braille_period):
            return self.msg_braille()

        elif self.time - self.last_scene > self.scene_period:
            return self.msg_scene()

    def msg_braille(self):
        self.last_braille = self.time
        return ""

    def msg_scene(self):
        self.last_scene = self.time
        guide_list = []
        for el in self.yolo:
            if el.name not in self.target_obs:
                continue
            if el.confidence < YOLO_THRES[el.name]:
                continue
            if el["depth"] < self.normal_range:  # 라이다가 없으면 물체를 전부 다 읽음
                guide_list.append(el)

        if self.priority == ReadPriority.LEFT_TO_RIGHT:
            sorted_obj = sorted(
                guide_list, key=lambda x: x["xc"], reverse=False
            )
        elif self.priority == ReadPriority.RIGHT_TO_LEFT:
            sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=True)
        elif self.priority == ReadPriority.NEAR_TO_FAR:
            sorted_obj = sorted(
                guide_list, key=lambda x: x["depth"], reverse=False
            )

        return obj2str(sorted_obj)


def obj2str(yolo: List[DetectorObject], warning: bool = False):
    msg = ""
    for el in yolo:
        korean_name = YOLO_NAME_TO_KOREAN[el["name"]]
        append = korean_name + " 충돌주의 " if warning else "" + korean_name + " "
        msg = msg + append
    return msg


def update_target_obs(target_obs_num):
    target_obs = []
    if target_obs_num & 1:
        for el in YOLO_OBS_TYPE["MOVING"]:
            target_obs.append(el)
    if target_obs_num & 2:
        for el in YOLO_OBS_TYPE["STATIC"]:
            target_obs.append(el)
    if target_obs_num & 4:
        for el in YOLO_OBS_TYPE["BRAILLE"]:
            target_obs.append(el)
    if target_obs_num & 8:
        for el in YOLO_OBS_TYPE["BUS_STOP"]:
            target_obs.append(el)
    if target_obs_num & 16:
        for el in YOLO_OBS_TYPE["OTHERS"]:
            target_obs.append(el)
    return target_obs


def temp():
    # Warning mesg generating based on YOLO
    if DEV_YOLO_BASED_WARNING:
        warning_obj = []
        for el in sorted_obj:
            if el["depth"] < 2:
                warning_obj.append(el)
        warning_msg = obj2str(warning_obj, True)

    # Warning mesg by depth map
    else:
        dist_map = (255 - depth_map) * ALPHA_TO_RANGE
        np.savetxt("output.txt", dist_map, fmt="%1.3f")
        warning_msg = ""

    logger.info(
        f"User trigger description: {user_trigger_msg}, Warning message: {warning_msg}"
    )
    return user_trigger_msg, warning_msg
