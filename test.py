import os
import glob
import cv2
import numpy as np
import math as m

""" Experimenting planar perspective transformation """

def drawPt(img,pt,color):
	cv2.circle(img, pt, 3, color, -1)

def ptToMat ((x,y)):
	return np.array([[x],[y],[1]])

def matToPt (ptMat):
	return (int(ptMat[0][0]/ptMat[2][0]), int(ptMat[1][0]/ptMat[2][0]))

# translation matrix
def tl(dx,dy):
	return np.array([[1, 0, dx],
					 [0, 1, dy],
					 [0, 0,  1]])

def rt(angle):
	p = m.pi
	a = m.radians(angle)
	return np.array([[  m.cos(a), m.sin(a),  0 ],
					 [ -m.sin(a), m.cos(a),  0 ],
					 [     0    ,    0    ,  1 ]])

def transform(pt, mat):
	return matToPt(np.dot(mat, ptToMat(pt)))


""" ====================== Main code ====================== """
img = np.zeros((500,1000,3), np.uint8) 
# cv2.rectangle(img,pt1,pt2,(0,255,0),3)
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
white = (255,255,255)


o = (0,0)
pts1 = [(150,0),(160,0),(170,0),(170,10),(170,20),(160,20),(150,20),(150,10)]
dx = 300; dy = 50
pts2 = [transform(pt, tl(dx,dy)) for pt in pts1]
pts3 = [transform(pt, rt(60)) for pt in pts1]

# translate the cordinate
dx = 200
dy = 250
o = transform(o, tl(dx,dy))
pts1 = [transform(pt, tl(dx,dy)) for pt in pts1]
pts2 = [transform(pt, tl(dx,dy)) for pt in pts2]
pts3 = [transform(pt, tl(dx,dy)) for pt in pts3]

# Visualize
cv2.line(img,(o[0]-1000,o[1]),(o[0]+1000,o[1]),white,1)
cv2.line(img,(o[0],o[1]-1000),(o[0],o[1]+1000),white,1)
for pt1 in pts1: drawPt(img,pt1,green)
for pt2 in pts2: drawPt(img,pt2,red)
for pt3 in pts3: drawPt(img,pt3,red)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()