import math
import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os

@njit(cache=True)
def radial_blur(img, num=30):
    height, width, _ = img.shape
    center = (int(width / 2), int(height / 2))

    img_blur = np.copy(img)

    weight = np.linspace(1, 1 / num, num)
    weight = weight / weight.sum()

    for y in range(height):
        for x in range(width):
            r = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            angle = math.atan2(y - center[1], x - center[0])

            tmp = [0, 0, 0]
            for i in range(num):
                new_r = max(0, r - i)
                new_x = int(new_r * math.cos(angle) + center[0])
                new_y = int(new_r * math.sin(angle) + center[1])

                new_x = min(new_x, width-1)
                new_y = min(new_y, height-1)

                tmp[0] += img[new_y, new_x, 0] * weight[i]
                tmp[1] += img[new_y, new_x, 1] * weight[i]
                tmp[2] += img[new_y, new_x, 2] * weight[i]

            img_blur[y, x, 0] = int(tmp[0])
            img_blur[y, x, 1] = int(tmp[1])
            img_blur[y, x, 2] = int(tmp[2])

    return img_blur

# 可视化权重矩阵
num = 30
weight = np.linspace(1, 1 / num, num)
weight = weight / weight.sum()

# plt.figure(figsize=(10, 6))
# plt.plot(weight, marker='o')
# plt.title('Radial Blur Weights')
# plt.xlabel('Index')
# plt.ylabel('Weight')
# plt.grid(True)
# plt.show()

if __name__=='__main__':
    for img_name in os.listdir('./HR'):
        if img_name[-3:]!='png' and img_name[-3:]!='jpg':
            pass
        img = cv2.imread(img_name)
        # blurred_img = radial_blur(img, num=50)


        height, width = img.shape[:2]
        downsampled_img = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('DIPDKP_lr_x2/'+img_name, downsampled_img)
