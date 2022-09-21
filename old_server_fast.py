import sys
sys.path.append("./yolov7/")
from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
from StateMachine import StateMachine
#from model import get_model
import os, io, time, json, easydict
from loguru import logger

import cv2
import numpy as np
import torch
# import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#from yolov7.yolov7_integrated import get_model  # 클래스 형식으로 만들기가 어려웠음 잘하는 사람 도와주길 바람

app = FastAPI()
stateMachine = StateMachine()


##### 모델 불러오기 #####
# model = get_model("wesee7_3.pt")
model_name = 'wesee7_fin.pt'

opt = easydict.EasyDict({'agnostic_nms':False, 'augment':True, 'classes':None, 'conf_thres':0.25, 'device':'cpu', 
                            'exist_ok':False, 'img_size':640, 'iou_thres':0.45, 'name':'exp', 'view_img':False,
                            'no_trace':False, 'nosave':False, 'project':'runs/detect', 'save_conf':True, 'save_txt':True,
                            'source':'','update':False, 'weights':[model_name]})

source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

#Options
logger.info(opt)

# Initialize
# set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

# Directories
save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
(save_dir).mkdir(parents=True, exist_ok=True)  # make dir
logger.info(f"Save dir is {save_dir}\n")
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, opt.img_size)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# BGR
colors[0] = [235,143,67] #Crosswalk
colors[1] = [0,0,255] #R_Signal
colors[2] = [0,255,0] #G_Signal
colors[3] = [0,212,255] #Braille

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

##### 모델 불러오기 완료 #####



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test")
async def test():
    return {"message": "test"}

@app.post("/upload")
async def file_upload(source: bytes = File(...), sequenceNo: int = 1):
    tick = time.time()

    im_file = source    
    im_id = sequenceNo

    # im = Image.open(io.BytesIO(im_file))
    encoded_img = np.fromstring(im_file, dtype = np.uint8)  # type : nparray
    img_cv = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)  
    
    logger.info(f"image recieved! size: {img_cv.size}, image conversion time: {time.time() - tick}")

    # model_tick = time.time()
    # results = model(im, size=640)
    # model_runtime = time.time() - model_tick
    # results.save(save_dir=f"run_imgs/{im_id}")
    # im.save(f"run_imgs/{im_id}/input.png", "PNG")
    # logger.info("model runtime: {}", model_runtime)

    # dataset = LoadSingleImage(img_cv, img_size=imgsz, stride=stride)  # 이부분 좀더 만져야됨
    dataset = LoadImages('image_sample/MP_KSC_007490.jpg', img_size=imgsz, stride=stride)
    result_dict={}  # sequenceNO : detection result

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        boxes=[]
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # p = Path(p)  # to Path
            s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
            
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            img_name = str(time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))+'-'+str(im_id)
            save_path = str(save_dir / img_name)+'.jpg'
            txt_path = str(save_dir / img_name)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(f"{int(cls)} {names[int(cls)]} {torch.tensor(xyxy).view(1, 4)[0].tolist()} {conf}")
                    coor = torch.tensor(xyxy).view(1, 4)[0].tolist()
                    box={}
                    box["xmin"]= coor[0]
                    box["ymin"]= coor[1]
                    box["xmax"]= coor[2]
                    box["ymax"]= coor[3]
                    box["confidence"]= round(float(conf),4)
                    box["class"]= int(cls)
                    box["name"]= names[int(cls)]

                    boxes.append(box)

                    if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(f'{int(cls)} {names[int(cls)]} {coor} {round(float(conf),5)}\n') if opt.save_conf else f.write(f'{int(cls)} {names[int(cls)]} {coor}\n')  # label format
                        # f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            
            print(f'Inference Done. ({t2 - t1:.3f}s)')

            results={}
            results["yolo"]=boxes
            print(results)
            result_dict[im_id] = results

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
    
    # result_dict = results.pandas().xyxy[0].to_dict(orient="records")
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