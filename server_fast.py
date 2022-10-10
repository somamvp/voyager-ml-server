from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
from StateMachine import StateMachine
import os, io, time, json, easydict, cv2
import numpy as np
from loguru import logger
from yolov7_wrapper import Detector

# 이미지 회전 여부
is_Rot = True

model_name = 'basic7_fin.pt' 
opt = easydict.EasyDict({'agnostic_nms':False, 'augment':True, 'classes':None, 'conf_thres':0.25, 'device':'cpu', 
                            'exist_ok':False, 'img_size':640, 'iou_thres':0.45, 'name':'exp', 'view_img':False,
                            'no_trace':False, 'nosave':False, 'project':'runs/detect', 'save_conf':True, 'save_txt':True,
                            'update':False, 'weights':[model_name]})

app = FastAPI()
stateMachine = StateMachine()
detector = Detector(opt)
result_dict={}  # 세션NO : detection result, 서비스 배포 시 여러장의 이미지를 한꺼번에 추론시키는 경우를 대비해 구축해놓았음.
                                            # 추후 메모리 누수 막기 위해 초기화시키는 알고리즘 필요

def bytes2cv(source: bytes):
    encoded_img = np.fromstring(source, dtype = np.uint8)  # type : nparray
    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    if is_Rot:
        RGBD = np.rot90(RGBD, 3)
    channel = RGBD.shape[2]
    # print(RGBD.shape)

    depth_cv = None
    if channel == 4:    # 4-channel image        
        # Premultiplied Resolving
        # RGBD = cv2.cvtColor(RGBD, cv2.COLOR_mRGBA2RGBA)
        depth_cv = (RGBD[:,:,3])
        img_cv = (RGBD[:,:,0:3])    
        img_cv = np.ascontiguousarray(img_cv, dtype=np.uint8)       
        print(f"Depth-integrated image recieved, depth value max: {depth_cv.max()}, min: {depth_cv.min()}")

    elif channel == 3:  # 3-channel image
        img_cv = RGBD

    return img_cv, depth_cv

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def test():
    return {"message": "test"}

@app.post("/upload")
async def file_upload(source: bytes = File(...), SESSION_NO: int = 1):
    # High-frequency Acting
    tick = time.time()
    high_freq = True

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(source)
    logger.info(f"image recieved! size: {rgb.size}, image conversion time: {time.time() - tick}")
    print(rgb.shape)

    # YOLO 추론
    result_dict[SESSION_NO] = detector.inference(rgb, SESSION_NO, depth_cv)  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리   


    logger.info(f'Inference Done. ({time.time()- tick:.3f}s)')
    logger.info("세션 아이디: {}", SESSION_NO)
    logger.info("발견된 물체: {}", [ result['name'] for result in result_dict[SESSION_NO]['yolo'] ])


    # StateMachine
    stateMachine.newFrame(result_dict[SESSION_NO]['yolo'])
    logger.info("사용자 안내: {}", stateMachine.guides)

    guide_enum = stateMachine.guides


    logger.info("/upload total runtime: {}", (time.time() - tick))
    # return high_freq, result_dict[SESSION_NO], guide_enum
    return guide_enum

@app.post("/update")
async def file_update(source: bytes = File(...), SESSION_NO: int = 1):
    # Low-frequency Acting
    tick = time.time()
    high_freq = False

    # 이미지 로딩
    rgb, depth_cv = bytes2cv(source)
    logger.info(f"image recieved! size: {rgb.size}, image conversion time: {time.time() - tick}")

    # YOLO 추론
    result_dict[SESSION_NO] = detector.inference(rgb, SESSION_NO, depth_cv)  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리   


    logger.info(f'Inference Done. ({time.time()- tick:.3f}s)')
    logger.info("세션 아이디: {}", SESSION_NO)
    logger.info("발견된 물체: {}", [ result['name'] for result in result_dict[SESSION_NO]['yolo'] ])

    logger.info("/update total runtime: {}", (time.time() - tick))
    return high_freq, result_dict[SESSION_NO]


@app.get("/start")
async def start():
    logger.info("--------restarting state machine----------")
    global stateMachine
    stateMachine = StateMachine()
    return ""