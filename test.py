import os
import glob
import cv2
import numpy as np

""" Experimenting planar perspective transformation """

img = np.zeros((1000,1000,3), np.uint8) 

pt1 = (200,300)
pt2 = (800,700)

cv2.rectangle(img,pt1,pt2,(0,255,0),3)

# img = cv2.resize(img, (1000,int(1000*r)))
# h,w = img.shape[:2]

# M = cv2.getRotationMatrix2D((w/2,h/2), 90, 1) # (center(x,y),angle,scale)
# img = cv2.warpAffine(img,M,(w,h))

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print os.listdir('./*.txt')
# print glob.glob('./*6.pdf')