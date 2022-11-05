import json
import time, os
import app.description as description
import easydict, cv2
import numpy as np
import pickle

from typing import Dict, Optional
from datetime import datetime
from fastapi import FastAPI, File, Form, Query
from loguru import logger
from pathlib import Path


from app.state_machine import Position, StateMachine
from app.tracking import TrackerWrapper
from app.wrapper_essential import (
    DetectorInference,
    DetectorObject,
    increment_path,
)
from app.description import ClockCycleStateActivator
from app.yolov7_wrapper import v7Detector
from app.voyager_metadata import (
    # YOLOV5_PT_FILE,
    YOLOV7_DESC_PT_FILE,
    YOLOV7_BASIC_PT_FILE,
    YOLO_NAME_TO_KOREAN,
    ALPHA_TO_RANGE,
)
from app.state_redis import rd
from app.state_saver import StateSaver


def getOpt(pt_file=""):
    opt = easydict.EasyDict(
        {
            "agnostic_nms": False,
            "augment": True,
            "classes": None,
            "conf_thres": 0.25,
            "device": "",
            "exist_ok": False,
            "img_size": 640,
            "iou_thres": 0.45,
            "name": "exp",
            "view_img": False,
            "no_trace": False,
            "nosave": False,
            "project": (
                # 도커 환경 여부에 따라 로그 디렉토리 변경
                "docker/runs/detect"
                if os.environ.get("docker") == "True"
                else "runs/detect"
            ),
            "save_conf": True,
            "save_txt": True,
            "update": False,
            "weights": [pt_file],
        }
    )
    return opt


opt = getOpt()

save_dir = Path(
    increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
)  # increment run
app = FastAPI()
stateMachine = StateMachine()
tracker = TrackerWrapper()

settings = {
    "scene_type": 2,
    "mode": 0,
    "scene_range": 6.0,
    # "warning_range": 1.1,
    # "braille_period": 15,
    # "scene_period": 30,
    # "warning_period": 5,
}
clockcyclestateactivator = ClockCycleStateActivator(settings, time.time())

Basicv7detector, Descv7detector = v7Detector(
    getOpt(YOLOV7_BASIC_PT_FILE), save_dir
), v7Detector(getOpt(YOLOV7_DESC_PT_FILE), save_dir)

logger.info(
    f"Using models: basic-{YOLOV7_BASIC_PT_FILE}, desc-{YOLOV7_DESC_PT_FILE}"
)
logger.info(f"Using basic settings: {settings}")

# 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
# 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요
result_dict: Dict[int, DetectorInference] = {}
desc_dict: Dict[int, DetectorInference] = {}


# 미터 단위의 Depth Map이 생성됨
def bytes2cv(source: bytes, is_rot: bool):
    encoded_img = np.fromstring(source, dtype=np.uint8)  # type : nparray
    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    if is_rot:
        RGBD = np.rot90(RGBD, 3)
    channel = RGBD.shape[2]

    depth_map = None
    if channel == 4:  # 4-channel image
        # Premultiplied Resolving
        RGBD = cv2.cvtColor(RGBD, cv2.COLOR_mRGBA2RGBA)
        depth_map = RGBD[:, :, 3]
        img_cv = RGBD[:, :, 0:3]
        img_cv = np.ascontiguousarray(img_cv, dtype=np.uint8)

        # 라이다값 맨위 3줄, 맨오른쪽 2줄 값이 이상함
        for i in range(3):
            for j in range(depth_map.shape[1]):
                depth_map[i][j] = depth_map[3][j]
        for j in range(2):
            for i in range(depth_map.shape[0]):
                depth_map[i][-1 - j] = depth_map[i][-3]

        logger.info(
            f"Depth integrated image recieved, depth value max: {depth_map.max()}, min: {depth_map.min()}"  # noqa : E501
        )

    else:  # normal image
        img_cv = np.ascontiguousarray(RGBD, dtype=np.uint8)

    return img_cv, depth_map


@app.get("/")
async def root(
    settings: str = Query(...),
):
    global clockcyclestateactivator
    settings = json.loads(settings)
    # logger.info(f"Updated settings: {settings}")
    clockcyclestateactivator = ClockCycleStateActivator(settings, time.time())
    return {"message": "Hello World"}


@app.get("/create")
async def create():
    # create csl

    state_machine = StateMachine(should_light_exist=None)
    tracker = TrackerWrapper()
    cs = ClockCycleStateActivator({}, time.time())

    redis_objects = StateSaver(state_machine, tracker, cs)

    return {"redis": redis_objects.stringify()}


@app.get("/test")
async def test():
    return {"message": "test"}


@app.post("/upload")
async def file_upload(
    source: bytes = File(...),
    sequence_no: int = Form(1),
    session_id: Optional[str] = Form(None),
    is_rot: bool = Form(True),
    gps_info: str = Form("{}"),
    cross_start: bool = Form(False),
    should_light_exist: Optional[bool] = Form(None),
):

    logger.info(f"session_id {session_id}; cross_start {cross_start}")

    if session_id is None:
        global stateMachine, tracker, clockcyclestateactivator

    if cross_start:
        logger.info(
            f"--------restarting state machine & tracker: light {should_light_exist}----------"
        )
        stateMachine = StateMachine(should_light_exist=should_light_exist)
        tracker = TrackerWrapper()
    elif session_id is not None:
        # 세션 아이디를 기반으로 레디스에 저장한 이전 값을 불러옴
        key = f"state:{session_id}"

        # redis에 저장된 값 파싱해서 쓰면 됌.
        # ex)"{'redis':{"state_machine": "b\\"\\\\x80\\\\x04\\\\x95X\\\\x01\\\\x00\\\\x00\\...}"
        data = rd.hget(key, "stateResult")
        string = json.loads(data)["redis"]

        state_saver = StateSaver.unstringify(string)

        stateMachine = state_saver.state_machine
        tracker = state_saver.tracker
        clockcyclestateactivator = state_saver.clock_activator

    # High-frequency Acting
    tick = time.time()
    log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_{session_id[:5]}_{sequence_no}"

    # 이미지 로딩
    rgb, depth_map = bytes2cv(source, is_rot)
    img_size = [rgb.shape[0], rgb.shape[1]]
    cv2.imwrite(f"{save_dir / log_str}.jpg", rgb)
    if depth_map is not None:
        cv2.imwrite(f"{save_dir / log_str}_depth.jpg", depth_map)
        depth_map = (255 - depth_map) * ALPHA_TO_RANGE

    logger.info(
        f"SESSION: {sequence_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
    )

    # YOLO 추론
    result_dict[session_id] = Basicv7detector.inference(
        source=rgb, im_id=sequence_no, save_name=log_str, depth_map=depth_map
    )
    logger.info(
        f"발견된 물체(횡단보도 안내): {[box.name for box in result_dict[session_id].yolo]}, time: {time.time() - tick}"
    )

    desc_dict[session_id] = Descv7detector.inference(
        source=rgb,
        im_id=session_id,
        save_name=log_str + "_d",
        depth_map=depth_map,
    )
    logger.info(
        f"발견된 물체(전체): {[box.name for box in desc_dict[session_id].yolo]}, time: {time.time() - tick}"
    )

    # Tracking & State Machine
    tracked_objects = tracker.update(
        result_dict[session_id].yolo,
        validate_zebra_cross=(img_size[0] // 2),
    )
    tracker.save_result(rgb, save_path=f"{save_dir / log_str}_tracking.jpg")

    position = None
    gps = json.loads(gps_info)
    if {"x", "y", "heading", "speed"}.issubset(gps):
        position = Position(gps["x"], gps["y"], gps["heading"], gps["speed"])
    stateMachine.newFrame(tracked_objects, position=position)
    guide_enum = stateMachine.guides
    # logger.info(f"트래킹, 스테이트머신 처리: time: {time.time() - tick}")

    # 안내 생성
    descrip_str, direction_warning_level = clockcyclestateactivator.inform(
        time.time(),
        desc_dict[session_id].yolo,
        stateMachine.is_now_crossing,
        stateMachine.crossroad_state,
        depth_map,
    )
    logger.info(
        f"횡단보도 안내: {guide_enum}, 일반 안내: {descrip_str}, time: {time.time() - tick}"
    )

    # 로깅
    logger.info("/upload total runtime: {}", (time.time() - tick))

    log_dict = {
        "is_depth": depth_map is not None,
        "rgb_shape": rgb.shape,
        "yolo1": [box.__dict__ for box in result_dict[session_id].yolo],
        "yolo2": [box.__dict__ for box in desc_dict[session_id].yolo],
        "position": position.__dict__ if position else {},
        "guide": guide_enum,
        "description": descrip_str,
        "direction_warning_level": direction_warning_level,
    }

    redis_objects = StateSaver(stateMachine, tracker, clockcyclestateactivator)
    result = [*range(3)]
    result[0] = {
        "guide": guide_enum,
        "yolo": "",  # 이제 이거 바꿔야됨
        "warning": descrip_str,
        "direction_warning_level": direction_warning_level,  # [2, 34, 96],
    }
    result[1] = {"logdict": log_dict}
    result[2] = {"redis": redis_objects.stringify()}

    return result


@app.get("/start")
async def start(should_light_exist: bool = Query(None)):
    logger.info(
        f"--------restarting state machine & tracker: should_light_exist {should_light_exist}----------"
    )
    global stateMachine, tracker
    stateMachine = StateMachine(should_light_exist=should_light_exist)
    tracker = TrackerWrapper()
    return ""


#############################
####### TESTING AREA ########


obstacle_on_crosswalk = ["car", "bus", "truck", "motorcycle", "bollard"]
height_metric = {
    "car": 100,
    "bus": 200,
    "truck": 150,
    "motorcycle": 100,
    "bollard": 100,
}  # 라이다 없는 경우 가깝다고 판단하는 기준 픽셀높이


def distance_test(yolo):
    guide_by_y, guide_by_depth = [], []
    sorted_obj_y, sorted_obj_depth = None, None

    # Guide by height
    for el in yolo:
        if el.name in obstacle_on_crosswalk and el.h > height_metric[el.name]:
            guide_by_y.append(el)
    sorted_obj_y = sorted(guide_by_y, key=lambda x: x["ymax"], reverse=True)

    # Guide by Lidar depth
    for el in yolo:
        if (
            el.name in obstacle_on_crosswalk
            and el["depth"] < settings["scene_range"]
        ):
            guide_by_depth.append(el)
    sorted_obj_depth = sorted(guide_by_depth, key=lambda x: x["depth"])

    return sorted_obj_y, sorted_obj_depth


# 간단한 테스트용으로 사용중
@app.post("/inform")
async def file_inform(
    source: bytes = File(...),
    session_no: int = Form(default=1, alias="sequenceNo"),
    is_rot: bool = Form(True),
    gps_info: str = Form("{}", alias="gpsInfo"),
):

    tick = time.time()
    log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_iSession{session_no}"

    # 이미지 로딩
    rgb, depth_map = bytes2cv(source, is_rot)
    logger.info(
        f"SESSION: {session_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
    )
    img_size = [rgb.shape[0], rgb.shape[1]]

    # YOLO 추론
    detected = Descv7detector.inference(
        source=rgb, im_id=session_no, save_name=log_str, depth_map=depth_map
    )

    sorted_yolo = sorted(detected.yolo, key=lambda x: x["ymax"], reverse=True)

    logger.info(
        f"발견된 물체(ymax순): {[(box.name, box.h, box.depth) for box in sorted_yolo if box.name in obstacle_on_crosswalk]}, time: {time.time() - tick}",
    )

    # Tracking & State Machine
    # tracked_objects = tracker.update(
    #     result_dict[session_no].yolo, validate_zebra_cross=(img_size[0] // 2)
    # )
    # tracker.save_result(rgb, save_path=f"{save_dir / log_str}_tracking.jpg")

    # position = None
    # gps = json.loads(gps_info)
    # if {"x", "y", "heading", "speed"}.issubset(gps):
    #     position = Position(gps["x"], gps["y"], gps["heading"], gps["speed"])
    # stateMachine.newFrame(tracked_objects, position=position)
    # guide_enum = stateMachine.guides

    # logger.info("사용자 안내: {}", guide_enum)

    # 전방 묘사
    sorted_obj_y, sorted_obj_depth = distance_test(detected.yolo)
    msg_y, msg_depth = " ".join(
        YOLO_NAME_TO_KOREAN[s.name] for s in sorted_obj_y
    ), " ".join(YOLO_NAME_TO_KOREAN[s.name] for s in sorted_obj_depth)

    scene_range = settings["scene_range"]
    logger.info("/upload total runtime: {}", (time.time() - tick))
    logger.info(f"판단기준 픽셀: {height_metric}, 거리기준: {scene_range }")
    logger.info(f"안내결과 - 높이기준 {msg_y}, 거리기준 {msg_depth}")

    # log_dict = {
    #     "is_depth": depth_map is not None,
    #     "rgb_shape": rgb.shape,
    #     "yolo_objects": [box.__dict__ for box in result_dict[session_no].yolo],
    #     "position": position.__dict__ if position else {},
    #     "guide": "",
    #     "description": descrip_str,
    #     "warning": warning_str,
    # }

    # print(json.dumps(log_dict), flush=True)
    return {
        "guide": [],  # No guide
        "yolo": f"높이기준 {msg_y}, 거리기준 {msg_depth}",
        "warning": "",
    }
