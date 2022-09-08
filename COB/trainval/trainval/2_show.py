import cv2
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
dict = sci.loadmat('./2008_000003.mat')
# print(dict)
img = dict["LabelMap"]
# print(img)
# print(img.shape)
# cv2.imshow("origin_mat_img",img)
# cv2.waitKey(0)

plt.imshow(img)
# plt.clim(0, 0.1)
plt.show()
# img = np.array(255*img, dtype="uint8")
# # cv2.imwrite('2.jpg', img )