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
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    NEAR_TO_FAR = 3


class ReadObstacle(IntEnum):
    # 안내를 원하는 객체타입을 이진마스킹 방식으로 지정 (chmod 777 하는거랑 같은 방식)
    # 0 이면 안내하는 객체 없음, 31이면 모든 객체 안내
    MOVING = 0b00001
    STATIC = 0b00010
    BRAILLE = 0b00100
    BUS_STOP = 0b01000
    OTHERS = 0b10000


# 이전 호출에서 입력받은 값을 기억하려고 하는데, 이렇게 그냥 전역변수로 써도 되나?
prev_target_obs_num: int = int(ReadObstacle.MOVING) + int(ReadObstacle.BRAILLE)
target_obs = [
    "person",
    "dog",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "wheelchair",
    "stroller",
    "kickboard",
    "Braille_Block",
]


def obj2str(yolo: List[DetectorObject], warning: bool = False):
    msg = ""
    for el in yolo:
        korean_name = YOLO_NAME_TO_KOREAN[el["name"]]
        append = korean_name + " 충돌주의 " if warning else "" + korean_name + " "
        msg = msg + append
    return msg


def update_target_obstacle(target_obs_num):
    target_obs.clear()
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


# 예상되는 문제1 너무 많은 객체 읽게 되는 경우
# 예상되는 문제2 적절한 depth range값을 찾기 어려움


# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7
def inform(
    depth_map,
    yolo: List[DetectorObject],
    priority: int = ReadPriority.LEFT_TO_RIGHT,
    target_obs_num: int = int(ReadObstacle.MOVING) + int(ReadObstacle.BRAILLE),
    normal_range: float = 4.0,
    warning_range: float = 2.0,
    img_size=[480, 640],
):

    if target_obs_num != prev_target_obs_num:
        update_target_obstacle(target_obs_num)

    guide_list = []
    for el in yolo:
        if el.name not in target_obs:
            continue
        if el.confidence < YOLO_THRES[el.name]:
            continue
        el["xc"] = (el["xmin"] + el["xmax"]) / 2
        el["yc"] = (el["ymin"] + el["ymax"]) / 2
        if (
            el["depth"]
            < normal_range
            # and abs(
            #     math.atan2(
            #         img_size[1] - el["ymax"], el["xc"] - (img_size[0] / 2)
            #     )
            # )
            # < math.pi / 4
        ):
            guide_list.append(el)

    if priority == ReadPriority.LEFT_TO_RIGHT:
        sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=False)
    elif priority == ReadPriority.RIGHT_TO_LEFT:
        sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=True)
    elif priority == ReadPriority.NEAR_TO_FAR:
        sorted_obj = sorted(guide_list, key=lambda x: x["depth"], reverse=False)
    else:
        logger.info("Description mode selected incorrectly")
    user_trigger_msg = obj2str(sorted_obj)

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
        print(
            f"Dist map: {dist_map.shape} {np.min(dist_map)} ~ {np.max(dist_map)}"
        )
        np.savetxt("output.txt", dist_map, fmt="%1.3f")
        warning_msg = ""

    logger.info(
        f"User trigger description: {user_trigger_msg}, Warning message: {warning_msg}"
    )
    return user_trigger_msg, warning_msg
