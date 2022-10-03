import sys, cv2
sys.path.append('./yolov7/')
import numpy as np
import time, json, easydict, torch
from pathlib import Path
from datetime import datetime
import random
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class Detector:
    def __init__(self, opt={}):
        self.opt = opt
        self.weights, self.save_txt, self.imgsz, self.trace = \
            opt.weights, opt.save_txt, opt.img_size, not opt.no_trace
        self.save_img = not opt.nosave # save inference images

        # Initialize
        # set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model

        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # logger.info(f"Save dir is {save_dir}\n")
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, opt.img_size)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # BGR
        self.colors[0] = [235,143,67] #Crosswalk
        self.colors[1] = [0,0,255] #R_Signal
        self.colors[2] = [0,255,0] #G_Signal
        self.colors[3] = [0,212,255] #Braille

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
       

    def inference(self, source, im_id, depth_cv=None):
        # tick = time.time()
        save_name = str(datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4])+f'_Session{im_id}'
        if type(source) is not str:
            dataset = LoadSingleImage(source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        
        # t0 = time.time()

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            boxes=[]
            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            for i, det in enumerate(pred):  # detections per image
                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # p = Path(p)  # to Path
                s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
                
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

                save_path = str(self.save_dir / save_name)+'.jpg'
                txt_path = str(self.save_dir / save_name)+'.txt'

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
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
                        box["name"]= self.names[int(cls)]
                        if depth_cv is not None:
                            scale = 3   # 클수록 협범위
                            box_w = coor[2]-coor[0]
                            box_h = coor[3]-coor[1]
                            hitbox = depth_cv[int(coor[0]+box_w/scale):int(coor[2]-box_w/scale), int(coor[1]+box_h/scale):int(coor[3]-box_h/scale)]
                            depth = np.mean(hitbox)
                            box['depth']= depth
                        else:
                            depth = 'NA' 

                        boxes.append(box)

                        # Save text
                        with open(txt_path, 'a') as f:
                            f.write(f'{int(cls)}\t %-18s %-30s\t %.05f\t {depth}\n'%(self.names[int(cls)], coor, round(float(conf),5)))

                        # Save image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                
                print(f'Inference Done. ({t2 - t1:.3f}s)')

                results={}
                results["yolo"]=boxes

                # Save results (image with detections)
                cv2.imwrite(save_path, im0)
                if depth_cv is not None:
                    Image.fromarray(depth_cv).save(f'{save_path}_depth.jpg')
                # print(f" The image with the result is saved in: {save_path}")
                
                return results


class LoadSingleImage:
    def __init__(self, img_cv, img_size=640, stride=32):
        self.cap = None
        self.img_cv = img_cv
        self.img_size = img_size
        self.stride = stride

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == 1:
            raise StopIteration

        img0 = self.img_cv
        self.count += 1
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        path = ''

        return path, img, img0, self.cap