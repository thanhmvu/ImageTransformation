import os
import glob
import cv2
import numpy as np

""" Experimenting planar perspective transformation """

def translate ((x,y), dx, dy ):
	# translation matrix
	mult = np.array([ [1, 0, dx],
					  [0, 1, dy],
					  [0, 0,  1] ])
	pt1 = np.array([[x],[y],[1]])
	pt2 = np.dot(mult,pt1)

	return (pt2[0][0],pt2[1][0])

def drawPt(img,pt,color):
	cv2.circle(img, pt, 3, color, -1)


""" ====================== Main code ====================== """
img = np.zeros((500,1000,3), np.uint8) 
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)

# cv2.rectangle(img,pt1,pt2,(0,255,0),3)

pt1 = (100,100)
drawPt(img,pt1,green)

dx = 100; dy = 0
pt2 = translate (pt1,dx,dy)
print pt2
drawPt(img,pt2,red)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()