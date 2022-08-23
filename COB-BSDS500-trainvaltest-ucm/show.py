import cv2
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
dict = sci.loadmat('./2018.mat')
img = dict["ucm2"]
print(img)
print(img.shape)
#cv2.imshow("origin_mat_img",img)
plt.imshow(img)
plt.clim(0, 0.1)
plt.show()
cv2.waitKey(0)
img = np.array(255*img, dtype="uint8")
# cv2.imwrite('2.jpg', img )