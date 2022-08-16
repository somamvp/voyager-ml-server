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
    "yolo": "test",
    "yolo_runtime": "test", 
    "guide": "1234",
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