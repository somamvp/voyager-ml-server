from flask import Flask, url_for, redirect, render_template, request
from PIL import Image
import os, io, time, json

# from model import get_model
from StateMachine import StateMachine

# model = get_model()

app = Flask(__name__)

stateMachine = StateMachine()

@app.route('/upload', methods=['POST'])
def file_upload():
  tick = time.time()

  im_file = request.files['img']    # request: multipart form data -> extract image data

  im_bytes = im_file.read()
  im = Image.open(io.BytesIO(im_bytes))
  print(f"image recieved! size: {im.size}, image conversion time: {time.time() - tick}")

  # model_tick = time.time()
  # results = model(im, size=640)
  # model_runtime = time.time() - model_tick
  # print(f"model runtime: {model_runtime}")
  
  # result_dict = results.pandas().xyxy[0].to_dict(orient="records")
  # print(f"object detected: {[ result['name'] for result in result_dict ]}")

  # stateMachine.newFrame(result_dict)

  return_dict = {
    "yolo": [
        {
            "xmin": 665.7411499023438,
            "ymin": 604.4212646484375,
            "xmax": 1724.0,
            "ymax": 967.1834106445312,
            "confidence": 0.9451523423194885,
            "class": 0,
            "name": "Zebra_Cross"
        },
        {
            "xmin": 1038.79052734375,
            "ymin": 420.12811279296875,
            "xmax": 1059.23779296875,
            "ymax": 472.7822570800781,
            "confidence": 0.841476321220398,
            "class": 1,
            "name": "R_Signal"
        }
    ],
    "yolo_runtime": 1.0199544429779053, 
    "guide": [
        "traffic light & crossroad detected! start guiding",
        "traffic light Red! stay."
    ],
    "state_machine": "1234"
  }

  return_json = json.dumps(return_dict)

  print(f"total runtime: {time.time() - tick}", flush=True)
  return return_json


@app.route('/start', methods=['GET'])
def start():
  print("--------restarting state machine----------")
  global stateMachine
  stateMachine = StateMachine()
  return ""

@app.route('/test', methods=['GET'])
def test():
  return "test"

if __name__ == '__main__':
  app.run('0.0.0.0')