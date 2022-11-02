# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}]
from enum import Enum, IntEnum
import numpy as np
import time
from typing import List
from loguru import logger
from app.voyager_metadata import YOLO_NAME_TO_KOREAN, YOLO_THRES, YOLO_OBS_TYPE
from app.wrapper_essential import DetectorObject

ALPHA_TO_RANGE = 20.0 / 255.0
GLOBAL_INTERVAL = 2  # sec
DEV_YOLO_BASED_WARNING = False


class ReadPriority(IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1
    NEAR_TO_FAR = 2


class ReadObstacle(IntEnum):
    # 안내를 원하는 객체타입을 이진마스킹 방식으로 지정 (chmod 777 하는거랑 같은 방식)
    # 0 이면 안내하는 객체 없음, 31이면 모든 객체 안내
    MOVING = 0b00001
    STATIC = 0b00010
    FLOOR = 0b00100
    BUS_STOP = 0b01000
    CROSSWALK = 0b10000


class Direction(IntEnum):
    NONE = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 4


obstacle_on_crosswalk = ["car", "bus", "truck", "motorcycle", "bollard"]
height_metric = {
    "car": 100,
    "bus": 200,
    "truck": 150,
    "motorcycle": 100,
    "bollard": 100,
}  # 라이다 없는 경우 가깝다고 판단하는 기준 픽셀높이

# Scene 문제점 : 너무 많은 객체 읽게 되는 경우
# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7


class ClockCycleStateActivator:
    # 한 세션 안에서 환경설정은 일정하다고 가정
    def __init__(
        self, settings, basetime: float = time.time(), img_size=[640, 480]
    ):
        self.settings = settings
        self.scene_target_number: int = settings["scene_type"]
        self.target_obs: List[str] = update_target_obs(self.scene_target_number)
        self.priority: int = settings["mode"]
        self.scene_range: float = settings["scene_range"]
        self.warning_range: float = settings["warning_range"]
        self.braille_period: float = settings["braille_period"]
        self.scene_period: float = settings["scene_period"]
        self.img_size = img_size

        self.yolo = None
        self.depth_map = None
        self.time = basetime

        # 직전 안내 시간
        self.last_scene: float = basetime
        self.last_braille: float = basetime
        self.last_global: float = basetime

        # 직전 안내 State
        self.prev_braille: int = 0  # 이거 사용안할듯
        self.prev_cross: int = 0
        self.curr_braille: int = 0
        self.curr_cross: int = 0

    def inform(
        self,
        nowtime: float,
        yolo: List[DetectorObject],
        is_now_crossing: bool,
        is_guiding_crossroad: bool,
        depth_cv=None,
    ):
        self.update(nowtime, yolo, depth_cv)

        if not self.available(nowtime):
            logger.info("Description Unavailable")
            return "", ""

        msg = self.inform_near()

        if msg != "":
            logger.info("Inform type: NEAR")
        elif is_now_crossing:
            logger.info("Inform type: NOW_CROSSING")
            msg += self.inform_on_cross()
            self.timer_reset(time.time())
        elif is_guiding_crossroad:
            logger.info("Inform type: WAITING LIGHT")
            msg += self.inform_waiting_light()
        else:
            if self.time - self.last_braille > self.braille_period:
                logger.info("Inform type: BRAILLE")
                msg += self.msg_braille()

            elif self.time - self.last_scene > self.scene_period:
                logger.info("Inform type: SCENE")
                msg += self.msg_scene()
            else:
                logger.info("Inform type: IDLE")
                return "", ""

        if msg != "":
            self.last_global = time.time()
        return msg, ""

    def available(self, time: float):
        return self.time - self.last_global > GLOBAL_INTERVAL

    def timer_reset(self, time: float):
        self.time = time
        self.last_scene = time
        self.last_braille = time

    def direct(self, xc) -> int:
        if xc > self.img_size[1] * 0.33 and xc < self.img_size[1] * 0.67:
            return Direction.CENTER
        elif xc <= self.img_size[1] * 0.33:
            return Direction.LEFT
        else:
            return Direction.RIGHT

    def update(
        self,
        time: float,
        yolo: List[DetectorObject],
        depth_cv: np.ndarray = None,
    ):
        self.yolo = yolo
        self.depth_map = (
            None if depth_cv is None else (255 - depth_cv) * ALPHA_TO_RANGE
        )

        self.time = time

        self.prev_braille = self.curr_braille
        self.prev_cross = self.curr_cross

        direction_list = []
        for el in self.yolo:
            if (
                el["name"] == "Braille_Block"
                and el["ymax"] > self.img_size[0] * 0.5
            ):  # 중간 높이 아래에 있는 점자블록만 안내
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

    def delay(self, delay: float = 0.5):
        self.last_scene += delay
        self.last_braille += delay

    def inform_near(self) -> str:
        if self.depth_map is None:
            logger.info("No near alert due to missing of depth_map")
            return ""
        # np.savetxt("output.txt", self.depth_map, fmt="%1.3f")
        # logger.info(f"warning_range: {self.warning_range}")

        h = self.img_size[0]
        w = self.img_size[1]
        alert_direction = 0
        high_map = self.depth_map[int(h * 0.05) : int(h * 0.5), :]
        vertical_sum = np.sum(high_map < self.warning_range, axis=0)
        start, mid1, mid2, end = (
            int(w * 0.05),
            int(w * 0.33),
            int(w * 0.66),
            int(w * 0.95),
        )
        zone_sums = [
            np.sum(vertical_sum[start:mid1]),
            np.sum(vertical_sum[mid1:mid2]),
            np.sum(vertical_sum[mid2:end]),
        ]
        logger.info(f"zone counts: {zone_sums}")
        for zone, cnt in enumerate(zone_sums):
            # print(f"Near count at {zone}, cnt: {cnt}")
            if cnt > 500:
                alert_direction += pow(2, zone)
        if alert_direction == 0:
            return ""
        else:
            return " 추돌주의" + dir2str(alert_direction)

    def inform_on_cross(self) -> str:
        guide_list = []
        if self.depth_map is None:
            for el in self.yolo:
                if (
                    el.name in obstacle_on_crosswalk
                    and el.h > height_metric[el.name]
                ):
                    guide_list.append(el)
            sorted_obj = sorted(
                guide_list, key=lambda x: x["ymax"], reverse=True
            )

        else:
            for el in self.yolo:
                if (
                    el.name in obstacle_on_crosswalk
                    and el["depth"] < self.scene_range
                ):
                    guide_list.append(el)
            sorted_obj = sorted(guide_list, key=lambda x: x["depth"])

        if sorted_obj:
            nearest = sorted_obj[0]
            return f" {nearest.name}" + dir2str(self.direct(nearest.xc))
        else:
            return ""

    def inform_waiting_light(self) -> str:
        if self.prev_cross != self.curr_cross:
            return " 횡단보도" + dir2str(self.curr_cross)
        else:
            return ""

    # def inform_regular(self) -> str:
    #     if self.time - self.last_braille > self.braille_period:
    #         return self.msg_braille()

    #     elif self.time - self.last_scene > self.scene_period:
    #         return self.msg_scene()
    #     else:
    #         return ""

    def msg_braille(self):
        self.last_braille = self.time
        return "점자블록" + dir2str(self.curr_braille, self.priority)

    def msg_scene(self):
        self.last_scene = self.time
        guide_list = []
        for el in self.yolo:
            if el.name not in self.target_obs:
                continue
            if el.confidence < YOLO_THRES[el.name]:
                continue
            if el["depth"] < self.scene_range:  # 라이다가 없으면 물체를 전부 다 읽음
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


def dir2str(direction: int, priority=ReadPriority.LEFT_TO_RIGHT):
    if direction == 0:
        return "미확인"

    msg_list: List[str] = []
    if direction >= 4:
        msg_list.append("우측")
    if (direction % 4) >= 2:
        msg_list.append("정면")
    if (direction % 2) >= 1:
        msg_list.append("좌측")

    if priority != ReadPriority.RIGHT_TO_LEFT:
        msg_list.reverse()
    msg = " ".join(str(s) for s in msg_list)
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
        for el in YOLO_OBS_TYPE["FLOOR"]:
            target_obs.append(el)
    if target_obs_num & 8:
        for el in YOLO_OBS_TYPE["BUS_STOP"]:
            target_obs.append(el)
    if target_obs_num & 16:
        for el in YOLO_OBS_TYPE["LIGHTS"]:
            target_obs.append(el)
    return target_obs


# def temp():
# Warning mesg generating based on YOLO
# if DEV_YOLO_BASED_WARNING:
#     warning_obj = []
#     for el in sorted_obj:
#         if el["depth"] < 2:
#             warning_obj.append(el)
#     warning_msg = obj2str(warning_obj, True)

# # Warning mesg by depth map
# else:
#     dist_map = (255 - depth_map) * ALPHA_TO_RANGE
#     np.savetxt("output.txt", dist_map, fmt="%1.3f")
#     warning_msg = ""

# logger.info(
#     f"User trigger description: {user_trigger_msg}, Warning message: {warning_msg}"
# )
# return user_trigger_msg, warning_msg
