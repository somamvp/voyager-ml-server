# RGBD webp 생성

from PIL import Image
import numpy as np
import cv2, cv

image_file = Image.open("sample_RGB.jpg").convert("RGB")
depth_file = Image.open("sample_D.jpg")

image_np = np.array(image_file)
depth_np = np.array(depth_file).reshape((640, 480, 1))

# print(image_np.shape, depth_np.shape)

concat_np = np.concatenate((image_np, depth_np), axis=2)


# print(concat_np.shape)

premul_cv = cv2.cvtColor(concat_np, cv2.COLOR_RGBA2mRGBA)
premul_img = Image.fromarray(premul_cv)
premul_img.save("merged.webp", "webp")

# image_file.save('converted.webp','webp')
