# RGB이미지의 바운딩 박스에 Depth값을 매칭시켜 Distance가 추가된 json 생성
from PIL import Image
import numpy as np
import cv2

image = Image.open('webp.webp')

with open('webp.webp', mode='rb') as f:
    # 이미지 로딩
    encoded_img = np.fromstring(f.read(), dtype = np.uint8)  # type : nparray

    RGBD = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    channel = RGBD.shape[2]
    if channel == 4:
        depth_cv = RGBD[:,:,3]
        img_cv = RGBD[:,:,0:3]
    else:
        img_cv = RGBD

    # img_cv = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    print(img_cv.shape)
    print(img_cv.shape[2])

    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # yolo모델 내부에서 알아서 RGB변환을 하기 때문에 서버단에선 필요없는 코드

    print(rgb.shape)

    Image.fromarray(rgb).show()
    # Image.fromarray(alpha).show()

    


# w, h = image.size
# pixels = image.convert('RGBA')
# raw = [[] for _ in range(4)]



