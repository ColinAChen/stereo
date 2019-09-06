import cv2
import stereo
from skimage.draw import line
import math
import queue

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

def test1():
	image0 = cv2.imread("frame0.jpg")
	image1 = cv2.imread("frame1.jpg")
	
	point = [330, 455]
	bounds = stereo.getSearchLine(point[0], point[1])
	print("Point:", point[0], point[1])
	print("Bounds:", bounds[0], bounds[1])
	
	image0[point[0]][point[1]] = (0, 0, 255)
	image1[330][432] = (0, 0, 255)
	
	print(stereo.realDistance(330, 455,  330, 432))
	
	if(False):
		cv2.imshow("window0", image0)
		cv2.imshow("window1", image1)
		cv2.waitKey(0)

def test2():
	image0 = cv2.imread("frame0.jpg")
	output = cv2.imread("frame0.jpg")

	'''	
	(239, 265): tape dispenser top
	(243, 265): tape dispenser bottom
	(258, 265): computer screen white edge
	'''
	point = (258, 265)
	RED = (0, 0, 255)
	colorObj(image0, point, RED, output)
	image0[point[0]][point[1]] = (0, 0, 255)
	
	cv2.imshow("0", image0)
	cv2.imshow("out", output)
	cv2.waitKey(0)

def colorObj(image, point, color, imOut):
	pixelQ = queue.Queue()
	pixelQ.put(point)

	pixelCount = 1
	while(not pixelQ.empty()):
		pixel = pixelQ.get()
		imOut[pixel[0]][pixel[1]] = color
		for r in range(-1, 2, 1):
			for c in range(-1, 2, 1):
				if((pixel[0]+r < 0) or (pixel[1]+c < 0)):
					break;
				if((pixel[0]+r > IMAGE_WIDTH) or (pixel[1]+c > IMAGE_HEIGHT)):
					break;
				if(tupleEquals(imOut[pixel[0]+r][pixel[1]+c], color)):
					continue
				
				if(matchPixels(image[pixel[0]+r][pixel[1]+c], image[pixel[0]][pixel[1]])):
					pixelQ.put((pixel[0]+r, pixel[1]+c))
					pixelCount += 1

		if(pixelCount % 10000 == 0):
			print(pixelCount, "pixels")
			cv2.imshow("out", imOut)
			cv2.waitKey(0)
			print("continuing")

def colorObjRecursive(image, point, color, imOut):
	for r in range(-1, 2, 1):
		for c in range(-1, 2, 1):
			if((point[0]+r < 0) or (point[1]+c < 0)):
				break;
			if((point[0]+r > IMAGE_WIDTH) or (point[1]+c > IMAGE_HEIGHT)):
				break;
			if(tupleEquals(imOut[point[0]+r][point[1]+c], color)):
				continue
			
			if(matchPixels(image[point[0]+r][point[1]+c], image[point[0]][point[1]])):
				imOut[point[0]+r][point[1]+c] = color
				colorObj(image, (point[0]+r, point[1]+c), color, imOut)

def matchPixels(pix1, pix2):
	thresh = 10

	a = (int(pix1[0]), int(pix1[1]), int(pix1[2]))
	b = (int(pix2[0]), int(pix2[1]), int(pix2[2]))
	if(dist(a, b) < thresh):
		return True
	else:
		return False

def dist(a, b):
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def tupleEquals(a, b):
	return a[0]==b[0] and a[1]==b[1] and a[2]==b[2]

test2()	