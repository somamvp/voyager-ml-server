# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}]
from enum import Enum, IntEnum
import numpy as np
import time
from typing import List
from loguru import logger
from app.voyager_metadata import YOLO_NAME_TO_KOREAN, YOLO_THRES, YOLO_OBS_TYPE
from app.wrapper_essential import DetectorObject

# ALPHA_TO_RANGE = 20.0 / 255.0
GLOBAL_INTERVAL = 2  # sec
# DEV_YOLO_BASED_WARNING = False


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


obstacle_on_crosswalk = ["car", "bus", "truck", "motorcycle", "bicycle", "kickbaord", "bollard"]
height_metric = {
    "car": 100,
    "bus": 200,
    "truck": 150,
    "motorcycle": 100,
    "bicycle": 100,
    "kickboard": 150,
    "bollard": 100,
}  # 라이다 없는 경우 가깝다고 판단하는 기준 픽셀높이

# Scene 문제점 : 너무 많은 객체 읽게 되는 경우
# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7


class ClockCycleStateActivator:
    # 한 세션 안에서 환경설정은 일정하다고 가정
    def __init__(
        self, settings, basetime: float = time.time(), img_size=[640, 480]
    ):
        # self.settings = settings
        self.scene_target_number: int = settings.get("scene_type", 1)
        self.target_obs: List[str] = update_target_obs(self.scene_target_number)
        self.priority: int = settings.get("mode", 0)
        self.scene_range: float = settings.get("scene_range", 6.0)
        # self.warning_range: float = settings.get("warning_range", 1.1)
        self.img_size = img_size
        self.sorted_obj = None
        self.last_global: float = basetime
        self.time = basetime
        
        self.prev_cross: int = 0
        self.curr_cross: int = 0
        self.curr_braille: int = 0
        self.lost_braille_cnt: int = 0
        self.warning_level: int = 0
        
        # 과거 사용코드
        # self.braille_period: float = settings.get("braille_period", 15)
        # self.scene_period: float = settings.get("scene_period", 30)
        # self.warning_period: float = settings.get("warning_period", 6)
        
        # self.depth_cnt: List[int] = None
        # self.last_warning: float = basetime
        # self.last_scene: float = basetime
        # self.last_braille: float = basetime

        # 직전 안내 State
        # self.prev_warning: int = 0
        # self.curr_warning: int = 0
        
        logger.info(settings)

    # 위험안내(warning) - yolobox기반, 비프음
    # 길안내 - 점자블록, 횡단보도 포함
    def inform(
        self,
        nowtime: float,
        yolo: List[DetectorObject],
        is_now_crossing: bool,
        is_guiding_crossroad: bool,
        depth_map:np.ndarray =None,
    ):
        self.update(nowtime, yolo, depth_map)
        
        # 위험도 계산
        warning_levels :List[int] = [0,0,0]
        for el in self.sorted_obj:
            idx = int(self.direct(el['xc']) / 2)
            warning_levels[idx] += pow(2.5, (6.0 - el['depth']))
            if warning_levels[idx] > 100:
                warning_levels[idx] = 100

        # 최근 안내가 너무 많을 때
        if not self.available(nowtime):
            logger.info("Inform type: Unavailable")
            return "", warning_levels

        msg = ""
        # 추돌주의 + 볼라드 (작동안함)
        # if ((self.prev_warning != self.curr_warning) and ((self.time - self.last_warning > self.warning_period) or (self.prev_warning == 0))) or /
        # (self.curr_bollard and self.prev_bollard != self.curr_bollard):
        #     msg = "볼라드 정면" if self.curr_bollard else ""
        #     msg += self.msg_warning()
        #     logger.info("Inform type: WARNING")

        # 횡단보도 건너는 중
        if is_now_crossing:
            logger.info("Inform type: NOW_CROSSING")
            msg = self.inform_on_cross()
            self.timer_reset(time.time())

        # 신호기다리는 중
        elif is_guiding_crossroad:
            logger.info("Inform type: WAITING LIGHT")
            msg = self.inform_waiting_light()

        # 점자블록 혹은 전방묘사
        # else:
            # if self.time - self.last_braille > self.braille_period:
            #     logger.info("Inform type: BRAILLE")
            #     msg = self.msg_braille()

            # elif self.time - self.last_scene > self.scene_period:
            #     logger.info("Inform type: SCENE")
            #     msg = self.msg_scene()
            # else:
            #     logger.info("Inform type: IDLE")
            #     return "", warning_levels

        self.last_global = time.time()
        return msg, warning_levels

    def available(self, time: float):
        return time - self.last_global > GLOBAL_INTERVAL

    def timer_reset(self, time: float):
        self.time = time
        self.last_scene = time
        self.last_braille = time

    def direct(self, xc) -> Direction:
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
        depth_map = None
    ):
        # self.yolo = yolo
        self.time = time

        # Update Zebra_Cross
        self.prev_cross = self.curr_cross
        sorted_yolo = sorted(yolo, key=lambda x: x["ymax"], reverse=True)
        cross = None
        for el in sorted_yolo:
            if el["name"] == "Zebra_Cross":
                cross = el
                break
        self.curr_cross = (
            Direction.NONE if cross is None else self.direct(cross["xc"])
        )

        # Update Warning Level
        guide_list = []
        for el in yolo:
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

        self.sorted_obj = sorted_obj
        
        # Update Braille
        if self.curr_braille == 0:
            self.lost_braille_cnt += 1
        else:
            self.lost_braille_cnt = 0
            
        direction_set = set()
        for el in yolo:
            if (
                el["name"] == "Braille_Block"
                and el["ymax"] > self.img_size[0] * 0.5
            ):  # 중간 높이 아래에 있는 점자블록만 안내
                direction_set.add(self.direct(el["xc"]))
                
        curr_braille = 0
        for num in direction_set:
            curr_braille += num
        self.curr_braille = curr_braille
        
    
    def OLD_update(
        self,
        time: float,
        yolo: List[DetectorObject],
        depth_map: np.ndarray = None,
    ):
        self.yolo = yolo
        self.depth_cnt = self.depth_counting(depth_map)

        self.time = time

        # self.prev_braille = self.curr_braille
        self.prev_cross = self.curr_cross
        self.prev_warning = self.curr_warning
        self.prev_bollard = self.curr_bollard
        
        # Update Bollard
        bollard_at_front = False
        for el in self.yolo:
            if (el['name' == "bollard"] and self.direct(el['xc']) == Direction.CENTER):
                bollard_at_front = True
                break
        self.curr_bollard = bollard_at_front
            
        # Update Braille
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

        # Update Zebra_Cross
        sorted_yolo = sorted(self.yolo, key=lambda x: x["ymax"], reverse=True)
        cross = None
        for el in sorted_yolo:
            if el["name"] == "Zebra_Cross":
                cross = el
                break
        self.curr_cross = (
            Direction.NONE if cross is None else self.direct(cross["xc"])
        )

        # Update warning
        self.curr_warning = self.update_warning()

    def delay(self, delay: float = 0.5):
        self.last_scene += delay
        self.last_braille += delay
        self.last_warning += delay


    def depth_counting(self, depth_map) -> List[int]:
        if depth_map is None:
            return None

        # np.savetxt("output.txt", self.depth_map, fmt="%1.3f")
        # logger.info(f"warning_range: {self.warning_range}")

        h = self.img_size[0]
        w = self.img_size[1]

        high_map = depth_map[int(h * 0.05) : int(h * 0.5), :]
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
        return zone_sums
    
    def update_warning(self) -> int:
        alert_direction = 0
        for zone, cnt in enumerate(self.depth_cnt):
            # print(f"Near count at {zone}, cnt: {cnt}")
            if cnt > 500:
                alert_direction += pow(2, zone)
        return alert_direction

    def inform_on_cross(self) -> str:
        guide_list = []
        if self.depth_cnt is None:
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
            return f" {YOLO_NAME_TO_KOREAN[nearest.name]}" + dir2str(
                self.direct(nearest.xc)
            )
        else:
            return ""

    def inform_waiting_light(self) -> str:
        if self.prev_cross != self.curr_cross:
            return " 횡단보도" + dir2str(self.curr_cross)
        else:
            return ""

    def msg_braille(self):
        # self.last_braille = self.time
        return "점자블록" + dir2str(self.curr_braille, self.priority)

    def msg_scene(self):
        # self.last_scene = self.time
        return obj2str(self.sorted_obj)
    
    
    def msg_warning(self):
        self.last_warning = self.time
        if self.curr_warning == 0:
            return ""
        msg = dir2str(self.curr_warning, self.priority)
        return "추돌주의" + msg


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
