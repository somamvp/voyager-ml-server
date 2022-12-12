import cv2
import numpy as np
import os, pathlib
from pathlib import Path
from natsort import natsorted

curdir = Path(__file__).parent
logdir = curdir / "app/runs/detect/221109_exp4"

dst = np.zeros((1280, 960, 3), dtype=np.uint8)
i = 0
for i, filename in enumerate(natsorted(os.listdir(logdir))):

    if "0d1ad" in filename:
        #  and ("_d_detection" in filename):
        if "_d_detection" in filename:
            dst[:640, :480, :] = cv2.imread(str(logdir / filename))
        if "_depth" in filename:
            dst[:640, 480:, :] = cv2.imread(str(logdir / filename))
        if "_tracking" in filename:
            dst[640:, :480, :] = cv2.imread(str(logdir / filename))

        # im = cv2.imread(str(logdir / filename))
        if i % 3 == 0:
            show = cv2.resize(dst, (640, 480))
            cv2.imshow("im", show)

            cv2.waitKey(30)
