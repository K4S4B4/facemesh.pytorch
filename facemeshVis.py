import matplotlib.pyplot as plt

import numpy as np
import cv2

#face_image = cv2.imread("mesh_map/mesh_map.jpg")[..., ::-1]

#all_points = np.load("mesh_map/facemesh_2d_points.npy")

#print(all_points)

#plt.figure(figsize=(20, 12))

#im = face_image.copy()
#for pt in all_points:
#    new_pt = np.array(pt)
#    new_pt = new_pt.astype(int)
#    im = cv2.circle(im, tuple(new_pt), 10, (100, 100, 200), 10)

#plt.imshow(im)


mesh = np.loadtxt('faceMeshPositions.csv', delimiter=',')
base_image = cv2.imread("facialMasks/base.png")

i = 0
for p in mesh:
    img = base_image.copy()
    p = p.astype(int)
    img = cv2.circle(img, tuple(p), 10, (255, 255, 255), -1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("facialMasks/facialMask"+str(i)+".jpg", img_gray)
    i = i + 1

    cv2.imshow("test", img)
    cv2.waitKey(1)
