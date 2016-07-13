import cv2
import numpy as np
import math
# import os
# import random

""" 
Python code to rotate an image without cropping
"""

""" Methods that add a black, square canvas surroung a given image """
def addCanvas(img, r):
  h,w = img.shape[:2]
  # Create a black canvas. to change color: canvas[:,:] = (255,0,0)
  canvas = np.zeros((r,r,3), np.uint8) 
  # Copy img onto canvas
  y1 = r/2 - h/2 ; y2 = r/2 + h/2
  x1 = r/2 - w/2 ; x2 = r/2 + w/2
  canvas[y1:y2,x1:x2] = img
  return canvas

"""
Method that returns the cordinates of 4 corners of a rotated rectangle.
The coordinates are relative to the center of the rectangle

The origin of the coordinate plane: O
    O -- x
    |
    y 
The rectangle has its the center at the origin with 4 vertices:
    TopLeft: A, TopRight: B
    BottomL: C, BottomR:  D
    
h - the height of the rectangle
w - the width of the rectangle
angRt - rotation angle in radian
"""
def getVertices(h,w,angRt):
  diag = math.sqrt(h*h + w*w) # Calculate OB
  
  # Calculate the angles between OB (top right) and Ox
  angTR1 = math.atan(float(h)/w)
  angTR2 = angTR1 + angRt
  
  x = int(math.cos(angTR2) * diag)
  y = - int(math.sin(angTR2) * diag)
  vertexTR = (x,y)
  vertexBL = (-x,-y)
  
  # Calculate the positive angle between OD (bottom right) and Ox
  angBR2 = angTR1 - angRt
  
  x = int(math.cos(angBR2) * diag)
  y = int(math.sin(angBR2) * diag)
  vertexBR = (x,y)
  vertexTL = (-x,-y)
  
  return [vertexTL, vertexTR, vertexBL, vertexBR]
 
"""
Methods returns the simple box (not the minimum area) surrounding a rotated rectangle given the 4 corners
"""
def surroundingBox(cornerTL, cornerTR, cornerBL, cornerBR):
  x = (cornerTL[0] + cornerBR[0])/2
  y = (cornerTL[1] + cornerBR[1])/2
  center = (x,y)
  
  minX = min(cornerTL[0], cornerTR[0], cornerBL[0], cornerBR[0])
  maxX = max(cornerTL[0], cornerTR[0], cornerBL[0], cornerBR[0])
  minY = min(cornerTL[1], cornerTR[1], cornerBL[1], cornerBR[1])
  maxY = max(cornerTL[1], cornerTR[1], cornerBL[1], cornerBR[1])
  
  w = maxY - minY + 1
  h = maxX - minX + 1
  
  return (center,h,w)
  
""" Method that rotates a given image by certain angle """
def rotate(img, angle, crop):
  h,w = img.shape[:2]

  if crop:
    # Calculate the rotation matrix then apply it
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1) # (center(x,y),angle,scale)
    rtImg = cv2.warpAffine(img,M,(w,h))    
    return rtImg
  
  else:
    # Calculate the size of the canvas
    r = int(math.sqrt(h*h + w*w)) + 1
    imgC = addCanvas(img,r*2)
    hC,wC = imgC.shape[:2]
    
    # Calculate the rotation matrix then apply it
    M = cv2.getRotationMatrix2D((wC/2,hC/2), angle, 1) # (center(x,y),angle,scale)
    rtImg = cv2.warpAffine(imgC,M,(wC,hC))
    
    relativeCorners = getVertices(h,w, math.radians(angle))
    center = (wC/2,hC/2)
    realCorners = [(corner[0]+center[0] , corner[1]+center[1]) for corner in relativeCorners]
#     print realCorners
#     print relativeCorners
    
#     box = surroundingBox(realCorners[0], realCorners[1], realCorners[2], realCorners[3])
    cv2.rectangle(rtImg, realCorners[0], realCorners[3], (255,0,0),3)
  
    return rtImg



# ======================================= Main =======================================
img = cv2.imread('0.jpg')

out = rotate(img,20,False)

cv2.imwrite('r2.jpg', out)

