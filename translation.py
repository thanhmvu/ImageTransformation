import cv2
import random
import math
import numpy as np
# from PIL import Image

img = cv2.imread("./test.jpg")

TITLE_RATIO = 0.2

def scaleAndTranslate(img, title):	
	""" This method generates training images from the ground images 
	by rescaling and translating the poster inside those images.
	The range of translation and scaling is hardcoded.
	
	@param title - list of 4 corners of the title
	@param img - the image to be transformed
	@return (imgT,title) - the transformed image and coordinates of the title' corners
	
	"""
	h1,w1 = img.shape[:2]
	# Generate n for scaling so that imageSize/ posterSize = n^2
	n = random.randint(10,15)*0.1 
	h2,w2 = (int(h1*n), int(w1*n))
	
	# Generate tx,ty for translation
	x1 = int(-0.2*w1) ; x2 = int(w2 - 0.8*w1)
	tx = random.randint(x1,x2)
	y1 = 0 ; y2 = int(h2 - 0.5*h1)
	ty = random.randint(y1,y2)
	
	# Transform image and calculate the new title
	M = np.float32([[1,0,tx],[0,1,ty]])
	stImg = cv2.warpAffine(img,M,(w2,h2))
	# title = [transformPoint(pt,M) for pt in title]
	# # refine out-of-bound points
	# for i,pt in enumerate(title):
	# 	x,y = pt
	# 	x = min(max(0,x), w2-1)
	# 	y = min(max(0,y), h2-1)
	# 	title[i] = (x,y)
	
	return (stImg,title)

for i in range(10):
	imgX, title = scaleAndTranslate(img,0)
	cv2.imshow("img",imgX); cv2.waitKey(0);



