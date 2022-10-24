# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}]
from enum import Enum
import math
from typing import List
from loguru import logger
from app.voyager_metadata import (
    YOLO_NAME_TO_KOREAN,
    YOLO_THRES,
)

from app.yolov7_wrapper import DetectorObject

RANGE_TO_ALPHA = 255 / 20


class ReadMode(Enum):
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    NEAR_TO_FAR = 3


def obj2str(yolo: List[DetectorObject], warning: bool = False):
    msg = ""
    for el in yolo:
        korean_name = YOLO_NAME_TO_KOREAN[el["name"]]
        append = korean_name + " 충돌주의 " if warning else "" + korean_name + " "
        msg = msg + append
    return msg


# 예상되는 문제1 너무 많은 객체 읽게 되는 경우
# 예상되는 문제2 적절한 depth range값을 찾기 어려움


# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7
def inform(
    depth_map,
    yolo: List[DetectorObject],
    mode: int = ReadMode.LEFT_TO_RIGHT,
    range: float = 4.0,
    img_size=[480, 640],
) -> dict:
    # 리턴값은 {"yolo":List, "warning":String}

    guide_list = []

    for el in yolo:
        if el.confidence < YOLO_THRES[el.name]:
            continue
        el["xc"] = (el["xmin"] + el["xmax"]) / 2
        el["yc"] = (el["ymin"] + el["ymax"]) / 2
        # if (
        #     el["depth"] < range
        #     and abs(
        #         math.atan2(
        #             img_size[1] - el["ymax"], el["xc"] - (img_size[0] / 2)
        #         )
        #     )
        #     < math.pi / 4
        # ):
        guide_list.append(el)

    if mode == ReadMode.LEFT_TO_RIGHT:
        sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=False)
    elif mode == ReadMode.RIGHT_TO_LEFT:
        sorted_obj = sorted(guide_list, key=lambda x: x["xc"], reverse=True)
    elif mode == ReadMode.NEAR_TO_FAR:
        sorted_obj = sorted(guide_list, key=lambda x: x["depth"], reverse=False)
    else:
        logger.info("Description mode selected incorrectly")

    # Warning mesg generating based on YOLO
    warning_obj = []
    for el in sorted_obj:
        if el["depth"] < 2:
            warning_obj.append(el)

    logger.info(
        f"User trigger description: {obj2str(sorted_obj)}, Warning message: {obj2str(warning_obj, True)}"
    )

    return obj2str(sorted_obj), obj2str(warning_obj, True)
