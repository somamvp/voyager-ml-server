# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}]
from enum import Enum, IntEnum
import numpy as np
import time
from typing import List
from loguru import logger
from app.voyager_metadata import YOLO_NAME_TO_KOREAN, YOLO_THRES, YOLO_OBS_TYPE, HEIGHT_METRIC
from app.wrapper_essential import DetectorObject
from app.state_machine import Position


GLOBAL_INTERVAL = 2  # sec


class ReadPriority(IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1
    # NEAR_TO_FAR = 2


class ReadObstacle(IntEnum):
    # 안내를 원하는 객체타입을 이진마스킹 방식으로 지정 (chmod 777 하는거랑 같은 방식)
    # 0 이면 안내하는 객체 없음, 31이면 모든 객체 안내
    MOVING = 0b00001
    STATIC = 0b00010
    FLOOR = 0b00100
    BUS_STOP = 0b01000


class Direction(IntEnum):
    NONE = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 4

OBSTACLE_ON_CROSSROAD = ["car", "bus", "truck", "motorcycle"]


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
        self.img_size: List[int,int] = img_size
        self.sorted_obj: List[DetectorObject] = None #안내대상만 읽는 순서대로 정렬한 것임
        self.last_global: float = basetime
        
        self.counter = 0
        self.toggle = True
        
        self.prev_position: Position = None
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
        is_depth: bool,
        position: Position = None,
        depth_map: np.ndarray = None,
    ):
        # 0.상태업데이트
        self.update(nowtime, yolo, is_depth, position)
        
        
        # 2.전방묘사버튼 문구
        descrip_msg = self.msg_scene()
        
        
        # 3.점자블록버튼 문구
        braille_msg = f"점자블록이 {dir2str(self.curr_braille, self.priority)}에 있습니다"
        
        
        # 1.강제안내문구
        
         # 최근 안내가 너무 많을 때
        # if not self.available(nowtime):
        #     logger.info("Inform type: Unavailable")
        #     return "", warning_levels

        msg = ""
        # 횡단보도 건너는 중
        if is_now_crossing:
            if self.toggle:
                self.toggle = False
                self.counter = 0
            else:
                self.counter += 1
                
            if counter == 3: #횡단보도 당 한번만 안내가 진행된다
                logger.info("Inform type: NOW_CROSSING")
                msg = self.inform_on_cross(yolo)
                # self.timer_reset(time.time())

        
        else:
            self.toggle = True
            self.counter = 0
            
            # 신호기다리는 중
            if is_guiding_crossroad:
                logger.info("Inform type: WAITING LIGHT")
                msg = self.inform_waiting_light()

            # State기반 점자블록 안내
            if self.lost_braille_cnt > 5 and self.curr_braille != 0:
                logger.info("Inform type: BRAILLE FOUND")
                msg = braille_msg


        if msg != "":
            self.last_global = time

 
        # 4.위험도
        warning_levels: List[int] = [0, 0, 0]
        if is_depth:
            for el in self.sorted_obj:
                idx = int(self.direct(el['xc']) / 2)
                warning_levels[idx] += max(int(119 - 19*el['depth']), 0)
                if warning_levels[idx] > 100:
                    warning_levels[idx] = 100
        else:
            for el in self.sorted_obj:
                idx = int(self.direct(el['xc']) / 2)
                standard = HEIGHT_METRIC[el.name]
                warning_levels[idx] += int((el['h'] - standard) / standard * 0.66)
                if warning_levels[idx] > 100:
                    warning_levels[idx] = 100
            
       

        return msg, descrip_msg, braille_msg, warning_levels

    def available(self, time: float):
        return time - self.last_global > GLOBAL_INTERVAL

    def timer_reset(self, time: float):
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
        is_depth: bool,
        position: Position
    ):
        # Update Zebra_Cross (가만히 있었으면 업데이트 안함)
        if self.prev_position and not position.similar_to(self.prev_position):
            self.prev_cross = self.curr_cross
            dist_sort = sorted(yolo, key=lambda x: x["ymax"], reverse=True)
            cross = None
            for el in dist_sort:
                if el["name"] == "Zebra_Cross":
                    cross = el
                    break
            self.curr_cross = (
                Direction.NONE if cross is None else self.direct(cross["xc"])
            )
        self.prev_position = position

        # Update Warning Level
        guide_list = []
        if is_depth:
            for el in yolo:
                if el.name not in self.target_obs or el.confidence < YOLO_THRES[el.name]:
                    continue
                if el["depth"] < self.scene_range:
                    guide_list.append(el)
        else:
            for el in yolo:
                if el.name not in self.target_obs or el.confidence < YOLO_THRES[el.name]:
                    continue
                if el['h'] >= HEIGHT_METRIC[el.name]:
                    guide_list.append(el)

        if self.priority == ReadPriority.LEFT_TO_RIGHT:
            sorted_obj = sorted(
                guide_list, key=lambda x: x["xc"], reverse=False
            )
        elif self.priority == ReadPriority.RIGHT_TO_LEFT:
            sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=True)
        # elif self.priority == ReadPriority.NEAR_TO_FAR:
        #     sorted_obj = sorted(
        #         guide_list, key=lambda x: x["depth"], reverse=False
        #     )

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
                and el["ymax"] > self.img_size[0] * 0.7
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

    def inform_on_cross(self, yolo) -> str:   # 처음 건널때 한번만 안내함
        guide_list = []
        cross = None
        
        #횡단보도 객체파악 및 안내대상 객체 가까운 순으로 정렬
        for el in yolo:
            if el.name == 'Zebra_Cross':
                cross = el
            elif el.name in OBSTACLE_ON_CROSSROAD:
                guide_list.append(el)
        guide_list = sorted(guide_list, key=lambda x: x['ymax'])
        if not cross or not guide_list:
            return ""
        
        #횡단보도와 겹쳐있는 물체 판별, 횡단보도 중심으로부터 너무 치우쳐진 곳에 있으면 부정확함
        critical_directions = [0,0] # 위험한 물체의 왼쪽 갯수, 오른쪽 갯수
        for el in guide_list:
            if el.ymax > cross.ymin:
                if el.xmax > (cross.xmin + cross.xc) / 2 or el.xmin < (cross.xmax + cross.xc) / 2:
                # 일단 지금은 아주 간단한 형태로 구현해놓음. 추후 사다리꼴 판별로 변경예정
                    if el.xc < cross.xc:
                        critical_directions[0] += 1
                    else:
                        critical_directions[1] += 1
        
        msg = ""
        if critical_directions[0] > 0 and critical_directions[1] > 0:
            msg = "양쪽"
        elif critical_directions[0] > 0:
            msg = "왼쪽"
        elif critical_directions[1] > 0:
            msg = "오른쪽"
        else:
            return ""
        
        return "횡단보도 " + msg + "에 차가 있으니 주의하세요"

    def inform_waiting_light(self) -> str:  # 위치가 변하지 않으면 안내 재생성하지 않도록 변경
        if self.prev_cross != self.curr_cross:
            return " 횡단보도" + dir2str(self.curr_cross)
        else:
            return ""

    # def msg_braille(self):
    #     self.last_braille = self.time
    #     return "점자블록" + dir2str(self.curr_braille, self.priority)

    def msg_scene(self):
        # self.last_scene = self.time
        now_direction = 0
        msg = ""
        for el in self.sorted_obj:
            if direct(el.xc) != now_direction:
                msg += dir2str(direct(el.xc))+" "
            msg += YOLO_NAME_TO_KOREAN[el.name]
        return msg
    
    
    # def msg_warning(self):
    #     self.last_warning = self.time
    #     if self.curr_warning == 0:
    #         return ""
    #     msg = dir2str(self.curr_warning, self.priority)
    #     return "추돌주의" + msg


def obj2str(yolo: List[DetectorObject]):
    msg = ""
    for el in yolo:
        korean_name = YOLO_NAME_TO_KOREAN[el["name"]]
        msg += korean_name + " "
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
