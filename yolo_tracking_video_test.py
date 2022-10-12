from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
import os, io, time, json, easydict, cv2, argparse
import numpy as np
from yolov7_wrapper import Detector
from StateMachine import StateMachine
from norfair import Video


model_name = 'basic7_fin.pt' 
opt = easydict.EasyDict({'agnostic_nms':False, 'augment':True, 'classes':None, 'conf_thres':0.25, 'device':'cpu', 
                            'exist_ok':False, 'img_size':640, 'iou_thres':0.45, 'name':'exp', 'view_img':False,
                            'no_trace':False, 'nosave':False, 'project':'runs/detect', 'save_conf':True, 'save_txt':True,
                            'update':False, 'weights':[model_name]})

stateMachine = StateMachine()
detector = Detector(opt)
result_dict={}  # 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
                                            # 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요


def test(video_path=None, process_every=30):

    video = Video(input_path=video_path)
    for i, frame in enumerate(video):
        if i % process_every != 0:
            continue

        SESSION_NO, rgb = i, cv2.rotate(frame, cv2.ROTATE_180)

        result_dict[SESSION_NO] = detector.inference(rgb, SESSION_NO, depth_cv=None)  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리   

        print("세션 아이디: {}", SESSION_NO)
        print("발견된 물체: {}", [ result['name'] for result in result_dict[SESSION_NO]['yolo'] ])

        # StateMachine
        stateMachine.newFrame(result_dict[SESSION_NO]['yolo'])
        print("사용자 안내: {}", stateMachine.guides)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("file", type=str, help="Video files to process")
    args = parser.parse_args()

    test(args.file)
