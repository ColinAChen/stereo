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
	image1 = cv2.imread("frame1.jpg")
	# should be image0 shape set to a default value
	output0 = cv2.imread("frame0.jpg")
	output1 = cv2.imread("frame1.jpg")

	'''	
	(239, 265): tape dispenser top
	(243, 265): tape dispenser bottom
	(258, 265): computer screen white edge
	(222, 230): top of brown box
	'''
	point0 = (239, 263)
	point1 = (239, 250)
	RED = (0, 0, 255)
	colorObj(image0, point0, RED, output0)
	colorObj(image1, point1, RED, output1)
	image0[point0[0]][point0[1]] = RED
	image1[point1[0]][point1[1]] = RED
	
	cv2.imshow("0", image0)
	cv2.imshow("out0", output0)
	cv2.imshow("1", image1)
	cv2.imshow("out1", output1)
	cv2.waitKey(0)

def test3():
	image0 = cv2.imread("frame0.jpg")
	out0 = cv2.imread("frame0.jpg")
	
	point = [200, 0]
	color = [0, 0, 255]
	while(point[1] < 640):
		c = (color[0], color[1], color[2])
		if(not tupleEquals(out0[point[0]][point[1]], c)):
			colorObj(image0, point, c, out0)
			if(color[0] != 255):
				color[0] += 1
			else:
				color[1] += 1
		point[1] += 1

	cv2.imshow("0", image0)
	cv2.imshow("out0", out0)
	cv2.waitKey(0)

BLACK = (0, 0, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
def test4():
	image0 = cv2.imread("frame0.jpg")
	out0 = cv2.imread("frame0.jpg")
	image1 = cv2.imread("frame1.jpg")
	out1 = cv2.imread("frame1.jpg")
	
	for r in range(len(out0)):
		for c in range(len(out0[r])):
			out0[r][c] = BLACK

	for r in range(len(out1)):
		for c in range(len(out1[r])):
			out1[r][c] = BLACK

	point0 = [222, 230]
	
	bounds = stereo.getSearchLine(point0[0], point0[1])
	drawLine(image1, bounds, RED)

	colorObjSquare(image0, point0, out0, 20, 20)

	
	image0[point0[0]][point0[1]] = RED
	cv2.imshow("0", image0)
	cv2.imshow("1", image1)
	cv2.imshow("out0", out0)
	cv2.imshow("out1", out1)
	cv2.waitKey(0)	

	cv2.imwrite("out0.jpg", out0) 
	cv2.imwrite("0.jpg", image0)
	

# only works with horizontal lines
def drawLine(image, bounds, color):
	for i in range(0, IMAGE_WIDTH):
		image[bounds[0][0]][i] = color

def colorObjSquare(image, point, imOut, size, thresh):
	origPt = (point[0], point[1])
	color = [0, 0, 5]
	objCount = 0
	pixCount = 0
	for pixR in range(0-int(size/2), int(size/2), 1):
		for pixC in range(0-int(size/2), int(size/2), 1):
			point[0] = origPt[0] + pixR
			point[1] = origPt[1] + pixC
			if(tupleEquals(imOut[point[0]][point[1]], BLACK)):
				colorObjSet(image, imOut, (point[0], point[1]), origPt[0]-int(size/2), origPt[0]+int(size/2), origPt[1]-int(size/2), origPt[1]+int(size/2), thresh, (color[0], color[1], color[2]))
				if(color[2] != 255):
					color[2] += 50
				elif(color[1] != 255):	
					color[1] += 50
				else:
					color[0] += 50
				objCount += 1
		
				'''
				cv2.imshow("out", imOut)
				cv2.waitKey(0)
				'''
			pixCount += 1	
			if(pixCount % 50 == 0):
					print(pixCount, "pixels out of", size**2, "completed")
	
	print(objCount, "objects detected")

def colorObjSet(image, imOut, point, top, bot, left, right, thresh, color):
	pixelQ = queue.Queue()
	pixelQ.put(point)
	while(not pixelQ.empty()):
		pixel = pixelQ.get()
		if(tupleEquals(imOut[pixel[0]][pixel[1]], BLACK)):	
			imOut[pixel[0]][pixel[1]] = color
		for r in range(-1, 2, 1):
			for c in range(-1, 2, 1):
				if((pixel[0]+r < top) or (pixel[1]+c < left)):
					break;
				if((pixel[0]+r > bot) or (pixel[1]+c > right)):
					break;
				if(not tupleEquals(imOut[pixel[0]+r][pixel[1]+c], BLACK)):
					continue
				if(r == 0 and c == 0):
					continue
				
				if(matchPixels(image[pixel[0]+r][pixel[1]+c], image[pixel[0]][pixel[1]], thresh)):
					pixelQ.put((pixel[0]+r, pixel[1]+c))

def colorObj(image, point, color, imOut, thresh):
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
				
				if(matchPixels(image[pixel[0]+r][pixel[1]+c], image[pixel[0]][pixel[1]], thresh)):
					pixelQ.put((pixel[0]+r, pixel[1]+c))
					pixelCount += 1

		if(pixelCount > 100000):
			break

		if(pixelCount % 100000 == -1):
			print(pixelCount, "pixels")
			cv2.imshow("out", imOut)
			cv2.waitKey(0)
			print("continuing")

def colorObjRecursive(image, point, color, imOut, thresh):
	for r in range(-1, 2, 1):
		for c in range(-1, 2, 1):
			if((point[0]+r < 0) or (point[1]+c < 0)):
				break;
			if((point[0]+r > IMAGE_WIDTH) or (point[1]+c > IMAGE_HEIGHT)):
				break;
			if(tupleEquals(imOut[point[0]+r][point[1]+c], color)):
				continue
			
			if(matchPixels(image[point[0]+r][point[1]+c], image[point[0]][point[1]], thresh)):
				imOut[point[0]+r][point[1]+c] = color
				colorObj(image, (point[0]+r, point[1]+c), color, imOut)

def matchPixels(pix1, pix2, thresh):
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

test4()
	