import os, shutil
import cv2, torch
import numpy as np

from typing import List
from pathlib import Path
from datetime import datetime
from torch import cudnn_convolution_transpose
from loguru import logger


from app.wrapper_essential import DetectorObject, DetectorInference
from datetime import datetime
from loguru import logger
from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    time_synchronized,
    TracedModel,
)

ALPHA_TO_RANGE = 20.0 / 255.0


class v7Detector:
    def __init__(self, opt, save_dir):
        self.opt = opt
        self.save_dir = save_dir
        self.weights, self.save_txt, self.imgsz, self.trace = (
            opt.weights,
            opt.save_txt,
            opt.img_size,
            not opt.no_trace,
        )
        self.save_img = not opt.nosave  # save inference images

        ## Initialize & Load model

        self.device = select_device(opt.device)
        self.half = (
            self.device.type != "cpu"
        )  # half precision only supported on CUDA

        self.model = attempt_load(
            self.weights, map_location=self.device
        )  # load FP32 model

        if self.trace:
            self.model = TracedModel(self.model, self.device, opt.img_size)

        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

        self.names = getattr(self.model, "module", self.model).names

        ## Directories

        project_path = Path(opt.project)
        if project_path.exists():
            for exp_dir in os.listdir(project_path):
                if len(os.listdir(f"{Path(opt.project)}/{exp_dir}")) == 0:
                    shutil.rmtree(f"{Path(opt.project)}/{exp_dir}")

        logger.info(f"Saving logs to {self.save_dir}...")
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Set box colors BGR
        self.colors = [
            [235, 143, 67],  # Zebra_Cross
            [0, 0, 255],  # R_Signal
            [0, 255, 0],  # G_Signal
            [0, 212, 255],  # Braille_Block
            [106, 181, 159],  # person
            [253, 55, 79],  # dog
            [131, 84, 79],  # tree
            [94, 96, 226],  # car
            [7, 104, 243],  # bus
            [229, 107, 98],  # truck
            [216, 151, 119],  # motorcycle
            [110, 41, 27],  # bicycle
            [103, 122, 117],  # none
            [84, 155, 200],  # wheelchair
            [240, 176, 4],  # stroller
            [28, 61, 11],  # kickboard
            [110, 161, 247],  # bollard
            [94, 163, 173],  # manhole
            [144, 200, 148],  # labacon
            [35, 32, 192],  # bench
            [225, 92, 63],  # barricade
            [82, 190, 117],  # pot
            [117, 110, 26],  # table
            [43, 8, 136],  # chair
            [54, 77, 102],  # fire_hydrant
            [184, 31, 24],  # movable_signage
            [247, 163, 39],  # bus_stop
        ]

    def getSavedir(self):
        return self.save_dir

    def inference(
        self, source, im_id, save_name: str, depth_map=None
    ) -> DetectorInference:

        # save_name ??? None?????? ????????? ??????
        save_path = self.save_dir / save_name if save_name is not None else None

        if type(source) is str:
            dataset = LoadImages(
                source, img_size=self.imgsz, stride=self.stride
            )
        else:
            dataset = LoadSingleImage(
                source.copy(), img_size=self.imgsz, stride=self.stride
            )

        for path, img_pad, img_orig, vid_cap in dataset:
            img_pad = torch.from_numpy(img_pad).to(self.device)
            img_pad = img_pad.float()  # uint8 to fp32
            img_pad /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_pad.ndimension() == 3:
                img_pad = img_pad.unsqueeze(0)

            boxes = []
            # Inference
            t1 = time_synchronized()
            pred = self.model(img_pad, augment=self.opt.augment)[0]
            # Apply NMS
            pred: List[torch.Tensor] = non_max_suppression(
                pred,
                self.opt.conf_thres,
                self.opt.iou_thres,
                classes=self.opt.classes,
                agnostic=self.opt.agnostic_nms,
            )
            t2 = time_synchronized()

            # logger.info(f"Inference Done. ({t2 - t1:.3f}s)")

            for det in pred:  # detections per image, (n, 6)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img_pad.shape[2:], det[:, :4], img_orig.shape
                ).round()

                img_save = img_orig.copy()

                if len(det) == 0:
                    continue

                # Write results
                for obj in reversed(det):
                    *coor, conf, cls = obj.tolist()
                    xc = (coor[0] + coor[2]) / 2
                    h = coor[3] - coor[1]

                    depth = -1
                    if depth_map is not None:
                        scale = 3  # ????????? ????????????, 2 ??????????????? ???
                        cutout_w = (coor[2] - coor[0]) / scale
                        cutout_h = (coor[3] - coor[1]) / scale
                        w_start, w_end = (
                            int(coor[0] + cutout_w),
                            int(coor[2] - cutout_w),
                        )
                        h_start, h_end = (
                            int(coor[1] + cutout_h),
                            int(coor[3] - cutout_h),
                        )

                        # (y,x) ???????????? ??????
                        hitbox = depth_map[h_start:h_end, w_start:w_end]
                        depth = round(np.mean(hitbox), 4)

                    box = DetectorObject(
                        *coor,
                        xc=xc,
                        h=h,
                        confidence=round(float(conf), 5),
                        cls=int(cls),
                        name=self.names[int(cls)],
                        depth=depth,
                    )
                    boxes.append(box)

                    # Save text
                    if save_path is not None:
                        with open(f"{save_path}.txt", "a") as f:
                            f.write(
                                f"{box.cls}\t {box.name:>15s} {str(coor):>30s}\t   {box.confidence:.05f}\t\t {depth}\t   {box.h}\n"
                            )

                    # Save image
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        coor,
                        img_save,
                        label=label,
                        color=self.colors[box.cls],
                        line_thickness=3,
                    )

            results = DetectorInference(yolo=boxes)

            # Save results
            if save_name is not None:
                cv2.imwrite(f"{save_path}_detection.jpg", img_save)

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
        path = ""

        return path, img, img0, self.cap
