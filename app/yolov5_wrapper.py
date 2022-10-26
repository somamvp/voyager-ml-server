import torch, cv2
import numpy as np
import app.yolov5.utils
from PIL import Image

from app.yolov5.models.common import AutoShape, DetectMultiBackend
from app.wrapper_essential import DetectorObject, DetectorInference
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using devide {DEVICE}")


class v5Detector:
    def __init__(self, pt_file, save_dir):
        self.model = AutoShape(
            DetectMultiBackend(weights=pt_file, device=torch.device(DEVICE))
        )
        self.save_dir = save_dir

    def inference(self, source: np.array, im_id, depth_cv=None):
        save_name = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{im_id}"
        save_path = self.save_dir / save_name

        image = Image.fromarray(source)
        results = self.model(image, size=640)
        results.save(save_dir=f"{save_path}")
        image.save(f"{save_path}.jpg", "JPG")

        return DetectorObject.from_dict(
            results.pandas().xyxy[0].to_dict(orient="records")
        )
