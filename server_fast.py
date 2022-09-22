from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
from StateMachine import StateMachine
import os, io, time, json, easydict, cv2
import numpy as np
from loguru import logger
from yolov7_wrapper import Detector

model_name = 'wesee7_fin.pt'
opt = easydict.EasyDict({'agnostic_nms':False, 'augment':True, 'classes':None, 'conf_thres':0.25, 'device':'cpu', 
                            'exist_ok':False, 'img_size':640, 'iou_thres':0.45, 'name':'exp', 'view_img':False,
                            'no_trace':False, 'nosave':False, 'project':'runs/detect', 'save_conf':True, 'save_txt':True,
                            'update':False, 'weights':[model_name]})

app = FastAPI()
stateMachine = StateMachine()
detector = Detector(opt)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def test():
    return {"message": "test"}

@app.post("/upload")
async def file_upload(source: bytes = File(...), sequenceNo: int = 1):
    tick = time.time()
    result_dict={}  # sequenceNO : detection result
    im_file = source    
    im_id = sequenceNo

    # 이미지 로딩
    encoded_img = np.fromstring(im_file, dtype = np.uint8)  # type : nparray
    img_cv = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)  
    
    logger.info(f"image recieved! size: {img_cv.size}, image conversion time: {time.time() - tick}")

    # YOLO 추론
    result_dict[im_id] = detector.inference(img_cv, im_id)
    # result_dict[im_id] = detector.inference('image_sample/MP_KSC_007490.jpg')
    logger.info(f'Inference Done. ({time.time()- tick:.3f}s)')
    


    # 결과 출력
    logger.info("발견된 물체: {}", [ result['name'] for result in result_dict[im_id]['yolo'] ])

    stateMachine.newFrame(result_dict[im_id]['yolo'])
    logger.info("사용자 안내: {}", stateMachine.guides)

    return_json = json.dumps(stateMachine.guides)

    logger.info("total runtime: {}", (time.time() - tick))
    return return_json


@app.get("/start")
async def start():
    logger.info("--------restarting state machine----------")
    global stateMachine
    stateMachine = StateMachine()
    return ""