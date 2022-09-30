# RGBA포맷에 실려온 이미지를 RGB와 D로 분리하고 Depth의 정보 손실이 없는지 확인

from PIL import Image
import numpy as np
import cv2

image = Image.open('webp.webp')

with open('webp.webp', mode='rb') as f:
    # 이미지 로딩
    encoded_img = np.fromstring(f.read(), dtype = np.uint8)  # type : nparray
    # print(encoded_img.shape)

    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    channel = RGBD.shape[2]
    print(f"Channel : {channel}")
    if channel == 4:
        depth_cv = RGBD[:,:,3]
        img_cv = RGBD[:,:,0:3]
    else:
        img_cv = RGBD

    # img_cv = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    # print(img_cv.shape)
    # print(img_cv.shape[2])

    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # yolo모델 내부에서 알아서 RGB변환을 하기 때문에 서버단에선 필요없는 코드

    print(rgb.shape)

    Image.fromarray(depth_cv).save("depth.jpg")
    # Image.fromarray(alpha).show()

    
