import cv2, time, requests
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, Form, Query

app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/call")
async def dummy_upload():
    return {"message": "Dummy well done"}


@app.post("/upload")
async def image_upload(source: bytes):
    f = request.files['file']
    f.save('./save_image/' + secure_filename(f.filename))
    # encoded_img = np.fromstring(source, dtype=np.uint8)
    # img = Image.fromarray(encoded_img)
    # img.save('recieved.jpg', 'jpg')
    # img.show()
    return {"message": "File call well done"}
