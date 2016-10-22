import math
import cv2
import random

TITLE_RATIO = 0.2

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def lightBlob(img, title):
	h,w = img.shape[:2]
	titleH, titleW = int(h*TITLE_RATIO), w

	xCenter = random.randint(0,titleW)
	yCenter = random.randint(0,titleH)
	R = random.randint(titleH/2,titleH*2)
	brightness = random.randint(0,11)*0.1
	
	# xCenter = titleW/2
	# yCenter = titleH/2
	# R = titleH
	# brightness = random.randint(0,11)*0.1

	mean = 0;
	sd = R/3;
	normalizedRatio = 1.0/normpdf(0,mean,sd)
	
	# adding the lighting blob by adjusting the brightness
	ltImg = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
	for i in range(-R,R):
		for j in range(-R,R):
			x = xCenter + i;
			y = yCenter + j;
			r = math.sqrt(i*i + j*j)
			if( x >= 0 and x < w and y >= 0 and y < h and r <= R):
			# if( x >= 0 and x < titleW and y >= 0 and y < titleH ):
				currL = ltImg[y][x][0]
				newL = currL * (1 + (normpdf(r,mean,sd))*normalizedRatio)
				newL = newL if newL < 255 else 255
				# newL = 255
				ltImg[y][x][0] = newL
	
	ltImg = cv2.cvtColor(ltImg,cv2.COLOR_LAB2RGB)
	return (ltImg,title)

img = cv2.imread("./test.jpg")

for i in range(10):
	n = random.randint(0,4);
	imgX = img
	for i in range(n):
		imgX, title = lightBlob(imgX,0)
	cv2.imshow("img",imgX); cv2.waitKey(0);
