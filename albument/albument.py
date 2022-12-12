import random
import cv2
from matplotlib import pyplot as plt

import albumentations as A

image = cv2.imread('1.jpg')

transform = A.Compose(
    [A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=4, p=1)],
)
# transform = A.Compose(
#     [A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)],
# )
random.seed(7)
transformed = transform(image=image)

cv2.imwrite('albumented.jpg', transformed['image'])
