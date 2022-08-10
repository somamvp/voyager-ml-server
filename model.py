import sys
sys.path.append("./yolov5")

import torch
from yolov5.models.common import DetectMultiBackend, AutoShape

def get_model(model_path="./best.pt"):

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"using devide {DEVICE}")

  model = AutoShape(
      DetectMultiBackend(weights=model_path, 
                        device=torch.device(DEVICE))
  )
  
  return model
