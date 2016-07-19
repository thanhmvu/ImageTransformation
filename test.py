import os
import glob
import cv2
import numpy as np
import math as m
from random import randint
import transformations as trans

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
p = m.pi


o = (0,0)
rec = []
h,w = (200,300)
rec.append([(0,0),(w,0),(w,h),(0,h)])

# # translate
# dx = 250; dy = 0
# rec.append([transform(pt, tl(dx,dy)) for pt in rec[0]])

# # rotate
# rec.append([transform(pt, rt(60)) for pt in rec[0]])

# # scale about origin
# rec.append([transform(pt, sc(2,2)) for pt in rec[0]])

# # combination
# a = m.radians(45)
# rec[0] = [transform(pt, tl(-50,-50)) for pt in rec[0]]
# rec.append([transform(pt, np.array([[ 2*m.cos(a)	, 2*m.sin(a)	, 50],
# 					 				[ -2*m.sin(a)	, 2*m.cos(a)	, 50],
# 					 				[ 0		, 0		, 1	]])) for pt in rec[0]])
# rec[0] = [transform(pt, tl(50,50)) for pt in rec[0]]



# rec.append([transform(pt, np.array([[ 0.9638,   -0.0960,   0],
# 					 				[ 0.2449,    1.3808,   0],
# 					 				[ -0.0001,   0.0013,   1]])) for pt in rec[0]])

# rec.append([transform(pt, np.dot(tl(150,-50),sc(2,2))) for pt in rec[0]])
# rec.append([transform(pt, np.dot(sc(2,2),tl(150,-50))) for pt in rec[0]])

# x = [transform(pt, tl(150,-50)) for pt in rec[0]]
# x = [transform(pt, sc(2,2)) for pt in x]
# rec.append(x)



# ================================= Orthogonal =====
# # center
# (x0,y0,z0) = (w/2,h/2,0)
# # R = max(w/2,h/2)

# ### find the plane
# # a point on the half sphere (S): (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = R ^2
# # x1 = randint(x0 - R, x0 + R)
# # tmp = int(m.sqrt(R*R - (x1-x0)**2))
# # y1 = randint(y0 - tmp, y0 + tmp)
# (x1,y1,z1) = (-20,100, 50)
# # z1 = int(m.sqrt(R*R - (x1-x0)**2 - (y1-y0)**2)) + z0 # z >= 0
# # print x1,y1,z1

# # vector n: (x1-x0, y1-y0, z1-z0)
# (Nx,Ny,Nz) = (x1-x0, y1-y0, z1-z0)
# # (P): Nx * (x-x1) + Ny * (y-y1) + Nz * (z-z1) = 0 = ax  + by + cz + d 
# (a,b,c,d) = (Nx, Ny, Nz, -(Nx*x1 + Ny*y1 + Nz*z1))
# # print (a,b,c,d)

# def projPoint((x2,y2,z2), (a,b,c,d)):
# 	# find the orthogonal projection of any point A onto that plane)
# 	# line through A and perpendicular to (P)
# 	# x = a*t + x2;  y = b*t + y2 ;  z = c*t + z2
# 	# a(at + x2) + b(bt + y2) + c(ct + z2) + d = 0
# 	t = float(-d -a*x2 -b*y2 -c*z2)/ (a*a + b*b + c*c)
# 	x = a*t + x2
# 	y = b*t + y2
# 	z = c*t + z2
# 	# print (x,y,z)
# 	return (x,y,z)

# projRec = []
# pt0 = rec[0][0]
# org2D = projPoint((pt0[0],pt0[1],0), (a,b,c,d))

# for pt in rec[0]:
# 	pt3D = projPoint((pt[0],pt[1],0), (a,b,c,d))

# 	tmp1 = np.array([[1,0,0],[0,1,0]])
# 	tmp2 = np.array([pt3D[0]-org2D[0],pt3D[1]-org2D[1],pt3D[2]-org2D[2]])
# 	dot = np.dot(tmp1, tmp2)
# 	# print dot

# 	pt2D = (int(dot[0]), int(dot[1]))
# 	projRec.append(pt2D)


# rec.append(projRec)

src_points = np.float32(rec[0])
dst_points = np.float32([(0,10),(w,10),(w,h-10),(0,h-10)])

# compute the transform matrix and apply it
M = cv2.getPerspectiveTransform(src_points,dst_points)
ptImg = cv2.warpPerspective(img,M,(w,h))
print ptImg

x = trans.rotation_matrix(m.pi/2, [0, 0, 1], [1, 0, 0])
print int( x[3][3])
print trans.scale_matrix(2)

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
	if x == rec[0]: c = red  
	elif x == rec[len(rec)-1]: c = blue
	else: c = green

	d = 2 if x == rec[0] else 2
	for i in range(len(x)): 
		if i == len(x)-1: cv2.line(img,x[i],x[0],c,d)
		else: cv2.line(img,x[i],x[i+1],c,d)


# cv2.imshow("img",ptImg)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()