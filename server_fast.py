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
    im_id = sequenceNo

    # 이미지 로딩
    encoded_img = np.fromstring(source, dtype = np.uint8)  # type : nparray
    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    if is_Rot:
        RGBD = np.rot90(RGBD, 3)
    channel = RGBD.shape[2]
    print(RGBD.shape)

    depth_cv = None
    if channel == 4:    # 4-channel image
        print("Depth-integrated image recieved")
        
        # Premultiplied Resolving
        # RGBD = cv2.cvtColor(RGBD, cv2.COLOR_mRGBA2RGBA)

        depth_cv = (RGBD[:,:,3])
        img_cv = (RGBD[:,:,0:3])   
        

        print(f"depth image max: {depth_cv.max()}, min: {depth_cv.min()}")
    elif channel == 3:  # 3-channel image
        img_cv = RGBD
    else:   # channel error
        print(f"Image Channel ERROR seqNO : {sequenceNo}")


    rgb = img_cv
    logger.info(f"image recieved! size: {rgb.size}, image conversion time: {time.time() - tick}")


    # YOLO 추론
    result_dict[im_id] = detector.inference(rgb, im_id, depth_cv)  # 리턴타입은 {'yolo':list of bbox}, 바운딩박스 자료형은 딕셔너리    
    logger.info(f'Inference Done. ({time.time()- tick:.3f}s)')
    


    # 결과 출력
    logger.info("세션 아이디: {}", im_id)
    logger.info("발견된 물체: {}", [ result['name'] for result in result_dict[im_id]['yolo'] ])

    stateMachine.newFrame(result_dict[im_id]['yolo'])
    logger.info("사용자 안내: {}", stateMachine.guides)

    guide_enum = json.dumps(stateMachine.guides)


    logger.info("total runtime: {}", (time.time() - tick))
    return guide_enum, result_dict[im_id]


@app.get("/start")
async def start():
    logger.info("--------restarting state machine----------")
    global stateMachine
    stateMachine = StateMachine()
    return ""