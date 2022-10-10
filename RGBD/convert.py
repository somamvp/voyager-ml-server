# RGBD webp 생성

from PIL import Image
import numpy as np

image_file = Image.open("Session1.jpg").convert('RGB')
depth_file = Image.open("Session1_depth.jpg")

image_np = np.array(image_file)
depth_np = np.array(depth_file).reshape((640, 480, 1))

# print(image_np.shape, depth_np.shape)

concat_np = np.concatenate((image_np, depth_np), axis=2)

# print(concat_np.shape)

concat_img = Image.fromarray(concat_np)
concat_img.save('merged.webp','webp')

# image_file.save('converted.webp','webp')
