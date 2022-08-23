import cv2
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
dict = sci.loadmat('./result.mat')
print(dict.keys())
side =  dict['late_sides']
print(side.shape)
print(dict['orientations'].shape)
print(dict['late_sides'])

# cv2.imread()
# print(img)
# print(img.shape)
# cv2.imshow("origin_mat_img",dict['late_sides'][0])
# plt.imshow(img)
# plt.clim(0, 0.1)
# plt.show()
# cv2.waitKey(0)
# img = np.array(255*img, dtype="uint8")
