import json
import time, os
import app.description as description
import easydict, cv2
import numpy as np

from typing import Dict, Optional
from datetime import datetime
from fastapi import FastAPI, File, Form
from loguru import logger
from pathlib import Path


from app.state_machine import Position, StateMachine
from app.tracking import TrackerWrapper
from app.wrapper_essential import (
    DetectorInference,
    DetectorObject,
    increment_path,
    IoU,
)
from app.description import ClockCycleStateActivator
from app.yolov7_wrapper import v7Detector
from app.voyager_metadata import (
    # YOLOV5_PT_FILE,
    YOLOV7_DESC_PT_FILE,
    YOLOV7_BASIC_PT_FILE,
    YOLO_NAME_TO_KOREAN,
)


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
    "scene_type": 3,
    "mode": 1,
    "scene_range": 6.0,
    "warning_range": 1.2,
    "braille_period": 15,
    "scene_period": 30,
}
clockcyclestateactivator = ClockCycleStateActivator(settings, time.time())
Basicv7detector, Descv7detector = v7Detector(
    getOpt(YOLOV7_BASIC_PT_FILE), save_dir
), v7Detector(getOpt(YOLOV7_DESC_PT_FILE), save_dir)

logger.info(
    f"Using models: basic-{YOLOV7_BASIC_PT_FILE}, desc-{YOLOV7_DESC_PT_FILE}"
)
logger.info(f"Using settings: {settings}")

# 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
# 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요
result_dict: Dict[int, DetectorInference] = {}
desc_dict: Dict[int, DetectorInference] = {}
# clock_state: Dict[int, ClockCycleStateActivator] = {}


def bytes2cv(source: bytes, is_rot: bool):
    encoded_img = np.fromstring(source, dtype=np.uint8)  # type : nparray
    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    # print(is_Rot)
    if is_rot:
        RGBD = np.rot90(RGBD, 3)
    channel = RGBD.shape[2]

    depth_cv = None
    if channel == 4:  # 4-channel image
        # Premultiplied Resolving
        RGBD = cv2.cvtColor(RGBD, cv2.COLOR_mRGBA2RGBA)
        depth_cv = RGBD[:, :, 3]
        img_cv = RGBD[:, :, 0:3]
        img_cv = np.ascontiguousarray(img_cv, dtype=np.uint8)

        # Depth map side pixel error resolve
        for i in range(3):
            for j in range(depth_cv.shape[1]):
                depth_cv[i][j] = depth_cv[3][j]
        for j in range(2):
            for i in range(depth_cv.shape[0]):
                depth_cv[i][-1 - j] = depth_cv[i][-3]

        logger.info(
            f"Depth integrated image recieved, depth value max: {depth_cv.max()}, min: {depth_cv.min()}"  # noqa : E501
        )

    else:  # normal image
        img_cv = np.ascontiguousarray(RGBD, dtype=np.uint8)

    return img_cv, depth_cv


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test")
async def test():
    return {"message": "test"}


@app.post("/upload")
async def file_upload(
    source: bytes = File(...),
    session_no: int = Form(default=1, alias="sequenceNo"),
    is_rot: bool = Form(True),
    gps_info: str = Form("{}", alias="gpsInfo"),
    settings: str = Form(...),
):
    # High-frequency Acting
    tick = time.time()
    global clockcyclestateactivator
    # clockcyclestateactivator.set(settings)
    log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{session_no}"

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(source, is_rot)
    img_size = [rgb.shape[0], rgb.shape[1]]
    cv2.imwrite(f"{save_dir / log_str}.jpg", rgb)
    if depth_cv is not None:
        cv2.imwrite(f"{save_dir / log_str}_depth.jpg", depth_cv)
    logger.info(
        f"SESSION: {session_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
    )

    # YOLO 추론
    result_dict[session_no] = Basicv7detector.inference(
        source=rgb, im_id=session_no, save_name=log_str, depth_cv=depth_cv
    )
    logger.info(
        f"발견된 물체(횡단보도 안내): {[box.name for box in result_dict[session_no].yolo]}, time: {time.time() - tick}"
    )

    desc_dict[session_no] = Descv7detector.inference(
        source=rgb,
        im_id=session_no,
        save_name=log_str + "_d",
        depth_cv=depth_cv,
    )
    logger.info(
        f"발견된 물체(전체): {[box.name for box in desc_dict[session_no].yolo]}, time: {time.time() - tick}"
    )

    # Tracking & State Machine
    tracked_objects = tracker.update(
        result_dict[session_no].yolo, validate_zebra_cross=(img_size[0] // 2)
    )
    tracker.save_result(rgb, save_path=f"{save_dir / log_str}_tracking.jpg")

    position = None
    gps = json.loads(gps_info)
    if {"x", "y", "heading", "speed"}.issubset(gps):
        position = Position(gps["x"], gps["y"], gps["heading"], gps["speed"])
    stateMachine.newFrame(tracked_objects, position=position)
    guide_enum = stateMachine.guides
    logger.info(f"트래킹, 스테이트머신 처리: time: {time.time() - tick}")

    # 안내 생성
    descrip_str, yolo_str = clockcyclestateactivator.inform(
        time.time(),
        desc_dict[session_no].yolo,
        stateMachine.is_now_crossing,
        stateMachine.is_guiding_crossroad,
        depth_cv,
    )
    logger.info(
        f"안내 생성됨. 횡단보도 안내: {guide_enum}, 일반 안내: {descrip_str}, time: {time.time() - tick}"
    )

    # 로깅
    logger.info("/upload total runtime: {}", (time.time() - tick))
    # logger.info("횡단보도 안내: {}", guide_enum)
    # logger.info("일반 안내: {}", descrip_str)
    log_dict = {
        "is_depth": depth_cv is not None,
        "rgb_shape": rgb.shape,
        "yolo_objects": [box.__dict__ for box in result_dict[session_no].yolo],
        "position": position.__dict__ if position else {},
        "guide": guide_enum,
        "description": descrip_str,
    }
    print(json.dumps(log_dict), flush=True)

    return {
        "guide": guide_enum,
        "yolo": "",  # 이제 이거 바꿔야됨
        "warning": descrip_str,
    }


# @app.post("/inform")
# async def file_inform(
#     source: bytes = File(...),
#     session_no: int = Form(default=1, alias="sequenceNo"),
#     is_rot: bool = Form(True),
#     gps_info: str = Form("{}", alias="gpsInfo"),
#     settings: str = Form(...),
# ):

#     settings = json.loads(settings)

#     tick = time.time()
#     log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_iSession{session_no}"

#     # 이미지 로딩
#     rgb, depth_cv = bytes2cv(source, is_rot)
#     logger.info(
#         f"SESSION: {session_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
#     )
#     img_size = [rgb.shape[0], rgb.shape[1]]

#     # YOLO 추론
#     result_dict[session_no] = Descv7detector.inference(
#         source=rgb, im_id=session_no, save_name=log_str, depth_cv=depth_cv
#     )

#     logger.info(
#         f"발견된 물체: {[box.name for box in result_dict[session_no].yolo]}, time: {time.time() - tick}",
#     )

#     # Tracking & State Machine
#     # tracked_objects = tracker.update(
#     #     result_dict[session_no].yolo, validate_zebra_cross=(img_size[0] // 2)
#     # )
#     # tracker.save_result(rgb, save_path=f"{save_dir / log_str}_tracking.jpg")

#     position = None
#     gps = json.loads(gps_info)
#     if {"x", "y", "heading", "speed"}.issubset(gps):
#         position = Position(gps["x"], gps["y"], gps["heading"], gps["speed"])
#     # stateMachine.newFrame(tracked_objects, position=position)
#     # guide_enum = stateMachine.guides

#     # logger.info("사용자 안내: {}", guide_enum)

#     # 전방 묘사
#     descrip_str, warning_str = description.inform(
#         depth_map=depth_cv,
#         yolo=result_dict[session_no].yolo,
#         img_size=img_size,
#         normal_range=4.0,
#     )

#     logger.info("/upload total runtime: {}", (time.time() - tick))
#     logger.info(
#         f"전방묘사 결과 - descrip_str: {descrip_str}, warning_str: {warning_str}"
#     )

#     log_dict = {
#         "is_depth": depth_cv is not None,
#         "rgb_shape": rgb.shape,
#         "yolo_objects": [box.__dict__ for box in result_dict[session_no].yolo],
#         "position": position.__dict__ if position else {},
#         "guide": "",
#         "description": descrip_str,
#         "warning": warning_str,
#     }

#     print(json.dumps(log_dict), flush=True)

#     return {
#         "guide": [],  # No guide
#         "yolo": descrip_str,
#         "warning": warning_str,
#     }


@app.get("/start")
async def start(should_light_exist: bool):
    logger.info("--------restarting state machine & tracker----------")
    global stateMachine, tracker
    stateMachine = StateMachine(should_light_exist=should_light_exist)
    tracker = TrackerWrapper()
    return ""
