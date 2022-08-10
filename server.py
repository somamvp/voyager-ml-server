from flask import Flask, url_for, redirect, render_template, request
from PIL import Image
import os, io, time, json

from model import get_model
from StateMachine import StateMachine

model = get_model()

app = Flask(__name__)

stateMachine = StateMachine()

@app.route('/upload', methods=['POST'])
def file_upload():
  tick = time.time()

  im_file = request.files['img']    # request: multipart form data -> extract image data

  im_bytes = im_file.read()
  im = Image.open(io.BytesIO(im_bytes))
  print(f"image recieved! size: {im.size}")

  model_tick = time.time()
  results = model(im, size=640)
  print(f"model runtime: {time.time() - model_tick}")
  
  result_dict = results.pandas().xyxy[0].to_dict(orient="records")
  print(f"object detected: {[ result['name'] for result in result_dict ]}")

  stateMachine.newFrame(result_dict)

  return_json = json.dumps(stateMachine.guides)

  print(f"total runtime: {time.time() - tick}", flush=True)
  return return_json


@app.route('/start', methods=['PUT'])
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