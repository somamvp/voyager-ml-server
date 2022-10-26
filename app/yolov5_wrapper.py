import torch
from app.yolov5.models.common import AutoShape, DetectMultiBackend

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using devide {DEVICE}")


model = AutoShape( 
    DetectMultiBackend(weights=src_pt, device=torch.device(DEVICE))
)
