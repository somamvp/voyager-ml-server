from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
from StateMachine import StateMachine
from model import get_model
import os, io, time, json

app = FastAPI()
stateMachine = StateMachine()

model = get_model("weseel_AL3.pt")

@app.get("//")
async def root():
    return {"message": "Hello World"}

@app.get("//test")
async def test():
    return {"message": "test"}

@app.post("//upload")
async def file_upload(img: bytes = File(...), sequenceNo: int = 1):
  tick = time.time()

  im_file = img    # request: multipart form data -> extract image data
  im_id = sequenceNo

  #im_bytes = im_file.read()
  im = Image.open(io.BytesIO(im_file))
  # print(f"image recieved! size: {im.size}, image conversion time: {time.time() - tick}")

  model_tick = time.time()
  results = model(im, size=640)
  model_runtime = time.time() - model_tick
  results.save(save_dir=f"run_imgs/{im_id}")
  im.save(f"run_imgs/{im_id}/input.png", "PNG")
  print(f"model runtime: {model_runtime}")
  
  result_dict = results.pandas().xyxy[0].to_dict(orient="records")
  print(f"발견된 물체: {[ result['name'] for result in result_dict ]}")

  stateMachine.newFrame(result_dict)
  print(f"사용자 안내: {stateMachine.guides}")

  # return_dict = {
  #   "yolo": result_dict,
  #   "yolo_runtime": model_runtime, 
  #   "guide": stateMachine.guides,
  #   "state_machine": ""
  # }

  return_json = json.dumps(stateMachine.guides)

  print(f"total runtime: {time.time() - tick}", flush=True)
  return return_json


@app.get("//start")
async def start():
    print("--------restarting state machine----------")
    global stateMachine
    stateMachine = StateMachine()
    return ""