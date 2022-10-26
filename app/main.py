import time, os

import easydict, cv2
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from fastapi import FastAPI, File, Form
from loguru import logger

import app.description as description
from app.state_machine import Position, StateMachine
from app.tracking import TrackerWrapper
from app.yolov7_wrapper import v7Detector, DetectorInference
from app.voyager_metadata import YOLO_PT_FILE


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
        "weights": [YOLO_PT_FILE],
    }
)

app = FastAPI()
stateMachine = StateMachine()
tracker = TrackerWrapper()
v7detector = v7Detector(opt)

# 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
# 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요
result_dict: Dict[int, DetectorInference] = {}


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
        # RGBD = cv2.cvtColor(RGBD, cv2.COLOR_mRGBA2RGBA)
        depth_cv = RGBD[:, :, 3]
        img_cv = RGBD[:, :, 0:3]
        img_cv = np.ascontiguousarray(img_cv, dtype=np.uint8)
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
    gps_x: Optional[float] = Form(None, alias="gpsX"),
    gps_y: Optional[float] = Form(None, alias="gpsY"),
    gps_heading: Optional[float] = Form(None, alias="gpsHeading"),
    gps_speed: Optional[float] = Form(None, alias="gpsSpeed"),
):
    # High-frequency Acting
    tick = time.time()
    log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{session_no}"

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(source, is_rot)
    logger.info(
        f"SESSION: {session_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
    )
    img_size = [rgb.shape[0], rgb.shape[1]]

    # YOLO 추론
    result_dict[session_no] = v7detector.inference(
        source=rgb, im_id=session_no, save_name=log_str, depth_cv=depth_cv
    )

    logger.info(
        f"발견된 물체: {[box.name for box in result_dict[session_no].yolo]}, time: {time.time() - tick}",
    )

    # Tracking & State Machine
    tracked_objects = tracker.update(
        result_dict[session_no].yolo, validate_zebra_cross=(img_size[0] // 2)
    )
    tracker.save_result(
        rgb, save_path=f"{v7detector.save_dir / log_str}_tracking.jpg"
    )

    gps_infos = [gps_x, gps_y, gps_heading, gps_speed]
    position = None if (None in gps_infos) else Position(*gps_infos)
    stateMachine.newFrame(tracked_objects, position=position)
    guide_enum = stateMachine.guides

    logger.info("사용자 안내: {}", guide_enum)

    # 전방 묘사
    descrip_str, warning_str = description.inform(
        depth_map=depth_cv,
        yolo=result_dict[session_no].yolo,
        img_size=img_size,
        normal_range=4.0,
    )

    logger.info("/upload total runtime: {}", (time.time() - tick))
    logger.info(f"descrip_str: {descrip_str}, warning_str: {warning_str}")

    return {
        "guide": guide_enum,
        # "yolo": [obj.__dict__ for obj in guide_dict["yolo"]],
        "yolo": descrip_str,
        "warning": warning_str,
    }

@app.post("/description")
async def file_upload(
    source: bytes = File(...),
    session_no: int = Form(default=1, alias="sequenceNo"),
    is_rot: bool = Form(True),
    gps_x: Optional[float] = Form(None, alias="gpsX"),
    gps_y: Optional[float] = Form(None, alias="gpsY"),
    gps_heading: Optional[float] = Form(None, alias="gpsHeading"),
    gps_speed: Optional[float] = Form(None, alias="gpsSpeed"),
):
    # Pulse Acting
    tick = time.time()
    log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{session_no}"

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(source, is_rot)
    logger.info(
        f"SESSION: {session_no} - image recieved! size: {rgb.shape}, image conversion time: {time.time() - tick}"
    )
    img_size = [rgb.shape[0], rgb.shape[1]]

    # YOLO 추론
    result_dict[session_no] = v7detector.inference(
        source=rgb, im_id=session_no, save_name=log_str, depth_cv=depth_cv
    )

    logger.info(
        f"발견된 물체: {[box.name for box in result_dict[session_no].yolo]}, time: {time.time() - tick}",
    )

    # Tracking & State Machine
    tracked_objects = tracker.update(
        result_dict[session_no].yolo, validate_zebra_cross=(img_size[0] // 2)
    )
    tracker.save_result(
        rgb, save_path=f"{v7detector.save_dir / log_str}_tracking.jpg"
    )

    gps_infos = [gps_x, gps_y, gps_heading, gps_speed]
    position = None if (None in gps_infos) else Position(*gps_infos)
    stateMachine.newFrame(tracked_objects, position=position)
    guide_enum = stateMachine.guides

    logger.info("사용자 안내: {}", guide_enum)

    # 전방 묘사
    descrip_str, warning_str = description.inform(
        depth_map=depth_cv,
        yolo=result_dict[session_no].yolo,
        img_size=img_size,
        normal_range=4.0,
    )

    logger.info("/upload total runtime: {}", (time.time() - tick))
    logger.info(f"descrip_str: {descrip_str}, warning_str: {warning_str}")

    return {
        "guide": guide_enum,
        # "yolo": [obj.__dict__ for obj in guide_dict["yolo"]],
        "yolo": descrip_str,
        "warning": warning_str,
    }



@app.post("/update")
async def file_update(
    source: bytes = File(...), session_no: int = 1, is_Rot=True
):
    # Low-frequency Acting
    tick = time.time()
    high_freq = False

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(is_Rot, source)
    logger.info(
        f"image recieved! size: {rgb.size}, image conversion time: {time.time() - tick}"
    )
    img_size = [rgb.shape[0], rgb.shape[1]]

    # YOLO 추론
    save_name = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{session_no}"
    result_dict[session_no] = v7detector.inference(
        rgb, session_no, save_name, depth_cv
    )  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리

    logger.info(f"Inference Done. ({time.time()- tick:.3f}s)")
    logger.info("세션 아이디: {}", session_no)
    logger.info(
        "발견된 물체: {}",
        [result["name"] for result in result_dict[session_no].yolo],
    )
    guide_dict = description.inform(
        result_dict[session_no].yolo, img_size=img_size
    )

    logger.info("/update total runtime: {}", (time.time() - tick))
    return {"high_freq": high_freq, "yolo": guide_dict}  # 정렬/솎아진 상태의 디텍션 정보


@app.get("/start")
async def start(should_light_exist: bool):
    logger.info("--------restarting state machine & tracker----------")
    global stateMachine, tracker
    stateMachine = StateMachine(should_light_exist=should_light_exist)
    tracker = TrackerWrapper()
    return ""
