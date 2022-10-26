import torch, cv2
from app.yolov5.models.common import AutoShape, DetectMultiBackend
from app.wrapper_essential import DetectorObject, DetectorInference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using devide {DEVICE}")


class v5Detector:
    def __init__(self, pt_file, save_dir):
        self.model = AutoShape(
            DetectMultiBackend(weights=pt_file, device=torch.device(DEVICE))
        )
        self.save_dir = save_dir

    def inference(self, source, im_id):
        save_name = f"{ datetime.now().strftime('%y%m%d_%H:%M:%S.%f')[:-4] }_Session{im_id}"
        save_path = self.save_dir / save_name

        results = (
            self.model(image, size=640)
            .pandas()
            .xyxy[0]
            .to_dict(orient="records")
        )
        cv2.imwrite(f"{save_path}.jpg", img_orig)
        cv2.imwrite(f"{save_path}_detection.jpg", img_save)

        return results
