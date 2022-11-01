# Depth map 의 값을 직접 출력

from PIL import Image
import numpy as np

image_file = Image.open("merged.webp")
image_np = np.array(image_file)
# print(image_np.shape)

np.savetxt("output.txt", image_np[:, :, 3], fmt="%1.2f")
