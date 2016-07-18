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

def transform(pt, mat):
	return matToPt(np.dot(mat, ptToMat(pt)))


# translation matrix
def tl(dx,dy):
	return np.array([[1, 0, dx],
					 [0, 1, dy],
					 [0, 0,  1]])

# rotation matric
def rt(angle):
	p = m.pi
	a = m.radians(angle)
	return np.array([[  m.cos(a), m.sin(a),  0 ],
					 [ -m.sin(a), m.cos(a),  0 ],
					 [     0    ,    0    ,  1 ]])

# scale about origin
def sc(W,H):
	return np.array([[ W, 0, 0],
					 [ 0, H, 0],
					 [ 0, 0, 1]])

""" ====================== Main code ====================== """
img = np.zeros((700,1000,3), np.uint8) 
# cv2.rectangle(img,pt1,pt2,(0,255,0),3)
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
white = (255,255,255)


o = (0,0)
rec = []
rec.append([(0,0),(100,0),(100,100),(0,100)])

# translate
dx = 250; dy = 0
rec.append([transform(pt, tl(dx,dy)) for pt in rec[0]])

# rotate
rec.append([transform(pt, rt(60)) for pt in rec[0]])

# scale about origin
rec.append([transform(pt, sc(2,2)) for pt in rec[0]])






# ===================================== Visualize ===================================== 
# translate the cordinate
dx = 200 	;	dy = 250
o = transform(o, tl(dx,dy))
for i,x in enumerate(rec):
	rec[i] = [transform(pt, tl(dx,dy)) for pt in x]

# Visualize
cv2.line(img,(o[0]-1000,o[1]),(o[0]+1000,o[1]),white,1)
cv2.line(img,(o[0],o[1]-1000),(o[0],o[1]+1000),white,1)
for x in rec:
	c = red if x == rec[0] else green
	d = 3 if x == rec[0] else 1
	for i in range(len(x)): 
		if i == len(x)-1: cv2.line(img,x[i],x[0],c,d)
		else: cv2.line(img,x[i],x[i+1],c,d)


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()