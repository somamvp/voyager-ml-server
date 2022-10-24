"""python yolo_tracking_video_test.py /home/soma1/MVP/voyager-juwon/tests/IMG_0265.MOV > video_test.log 2>&1"""

import time, argparse

import easydict, cv2
import numpy as np
from typing import Dict
from datetime import datetime
from loguru import logger
from norfair import Video

import app.description as description
from app.state_machine import StateMachine
from app.tracking import TrackerWrapper
from app.yolov7_wrapper import Detector, DetectorInference
from app.voyager_metadata import YOLO_PT_FILE

model_name = YOLO_PT_FILE
opt = easydict.EasyDict(
    {
        "agnostic_nms": False,
        "augment": True,
        "classes": None,
        "conf_thres": 0.25,
        "device": "cpu",
        "exist_ok": False,
        "img_size": 640,
        "iou_thres": 0.45,
        "name": "exp",
        "view_img": False,
        "no_trace": False,
        "nosave": False,
        "project": "runs/detect",
        "save_conf": True,
        "save_txt": True,
        "update": False,
        "weights": [model_name],
    }
)

stateMachine = StateMachine(use_gps=False)
tracker = TrackerWrapper()
detector = Detector(opt)

# 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
# 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요
result_dict: Dict[int, DetectorInference] = {}


def test(video_path=None, process_every=30):

    video = Video(input_path=video_path)
    for i, frame in enumerate(video):
        if i % process_every != 0:
            continue

        # if 500 < i < 4000:
        #     continue

        session_no, rgb = i, cv2.rotate(frame, cv2.ROTATE_180)
        log_str = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{session_no}"

        result_dict[session_no] = detector.inference(
            rgb, session_no, save_name=None, depth_cv=None
        )  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리

        logger.info("세션 아이디: {}", session_no)
        logger.info(
            "발견된 물체: {}",
            [result["name"] for result in result_dict[session_no]["yolo"]],
        )

        # Tracking & State Machine
        tracked_objects = tracker.update(
            result_dict[session_no].yolo,
            validate_zebra_cross=(rgb.shape[0] // 2),
        )

        stateMachine.newFrame(tracked_objects)
        guide_enum = stateMachine.guides

        logger.info("사용자 안내: {}", guide_enum)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("file", type=str, help="Video files to process")
    args = parser.parse_args()

    test(args.file, 15)
