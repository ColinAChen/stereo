import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import PIL
import argparse
import json

# real world cm height/width
realWidth = .86808
realHeight = .65106
# distance between focal point and frame
focalDistance = 0.625 # originally calculated as 1.915
focalX = realWidth/2
focalY = realHeight/2
# distance between focal points
frameDistance = 9
# pixel dimensions:
pixelWidth = 640
pixelHeight = 480

def pixelToReal(row,col):
	# return: 3D coordinates in cm of a point on the image plane
	cmx = realWidth * (col/pixelWidth)
	cmy = realHeight * (row/pixelHeight)
	return (focalDistance,cmx-focalX,focalY-cmy)

def realToPixel(cmx,cmy):
	# return: 2D cordinates of a pixel from 3D coordinates in cm
	col = (cmx + focalX)*pixelWidth/realWidth
	row = (focalY - cmy)*pixelHeight/realHeight
	return (round(row), round(col))

def pixelToRealFrame2(row, col):
	# return 3D coordinates in cm of point on second image plane
	cmx = realWidth * (col/pixelWidth)
	cmy = realHeight * (row/pixelHeight)
	return (focalDistance,cmx-focalX+frameDistance,focalY-cmy)

def realToPixelFrame2(cmx,cmy):
	# return: 2D cordinates on frame2 of a pixel from 3D coordinates in cm
	col = (cmx + focalX-frameDistance)*pixelWidth/realWidth
	row = (focalY - cmy)*pixelHeight/realHeight
	return (round(row), round(col))

def getDist(a, b):
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

firstFP = (0, 0, 0)
secondFP = (0, frameDistance, 0)	
def realDistance(firstR, firstC, secR, secC):
	# get first vector
	firstPt = pixelToReal(firstR, firstC)
	firstVec = (firstPt[0]-firstFP[0], firstPt[1]-firstFP[1], firstPt[2]-firstFP[2])
	
	# get second vector
	secPt = pixelToRealFrame2(secR, secC)
	secVec = (secPt[0]-secondFP[0], secPt[1]-secondFP[1], secPt[2]-secondFP[2])

	'''
	solve for intersections of eqs:
	<firstFP[0], firstFP[1], firstFP[2]> + t1<firstVec[0], firstVec[1], firstVec[2]>
	<secondFP[0], secondFP[1], secondFP[2]> + t2<secVec[0], secVec[1], secVec[2]>
	'''
	# solve system of eqs for t1 and t2	
	a = [firstVec[0], -1*secVec[0]]
	b = [firstVec[1], -1*secVec[1]]
	c = [firstVec[2], -1*secVec[2]]
	coefficients = np.array([a, b, c])
	constants = np.array([secondFP[0]-firstFP[0], secondFP[1]-firstFP[1], secondFP[2]-firstFP[2]])
	ans = np.linalg.lstsq(coefficients, constants)

	x = firstFP[0] + ans[0]*firstVec[0]
	y = firstFP[1] + ans[0]*firstVec[1]
	z = firstFP[2] + ans[0]*firstVec[2]

	return getDist((x[0], y[0], z[0]), firstFP)

# for working with tuples of the form (row, col)
ROW = 0
COL = 1

INF = 1000000000	# represents infinity
DOF = 0			# number of pixels to search above or below corresponding row when matching pixels

# takes tuple of two ints that has been cast to string and returns it as a tuple
def tup(strInput):
	commaIndex = 0
	length = len(strInput)
	for i, char in enumerate(strInput):
		if(char == ","):
			commaIndex = i

	num1 = int(strInput[1:commaIndex])
	num2 = int(strInput[commaIndex+2:len(strInput)-1])
	return (num1, num2)

# point is a tuple representing (row, col) of point on frame0
# function searches for a matching point on frame1
# dof is the degrees of freedom allowed to search away from the corresponding row
def matchPixels(frame0, frame1, point, window, dof):
	IMAGE_HEIGHT = frame0.shape[0]
	IMAGE_WIDTH = frame0.shape[1]
	
	# check for illegal input point
	if(point[ROW] < 0 or point[ROW] > IMAGE_HEIGHT):
		return (-1, -1)
	if(point[COL] < 0 or point[COL] > IMAGE_WIDTH):
		return (-1, -1)

	minDiff = INF
	match = (-1, -1)
	box0 = frame0[point[ROW]:point[ROW]+window, point[COL]:point[COL]+window]	# !!! Does not work with pixels lower than HEIGHT-window and WIDTH-window
	# iterate through search space in frame1
	for row in range(-1*dof, dof+1):
		row = row + point[ROW]
		if(row < 0):								# !!! Remove redundant loops (ex. multiple negative rows)
			row = 0
		if(row > IMAGE_HEIGHT-window):
			row = IMAGE_HEIGHT-window
		
		# matches window with target pixel in top left corner
		for col in range(0, frame1.shape[1]-window):
			box1 = frame1[row:row+window, col:col+window]
			boxDiff = sum(sum(abs(np.subtract(box0, box1))))
			if(boxDiff < minDiff):
				minDiff = boxDiff
				match = (row, col)
	return match

def matchFrames(frame0, frame1, window, dof):
	localDict = {}
	numPixels = 0

	# iterate through the frame
	for row in range(0, frame0.shape[0]-window):
		for col in range(0, frame0.shape[1]-window):
			point = (row, col)
			match = matchPixels(frame0, frame1, point, window, dof)
			localDict[str(point)] = str(match)
			numPixels += 1

			if(numPixels % 1000 == 0):
				print("Pixels completed:", numPixels)	
	return localDict

class CommandLine:
	def __init__(self, inOpts=None) :
		'''
		Implement a parser to interpret the command line argv string using argparse.
		'''

		import argparse
		self.parser = argparse.ArgumentParser(description = 'Choose to view real time video or saved images', 
												epilog = 'Stereo cameras to estimate depth', 
												add_help = True, #default is True 
												prefix_chars = '-', 
												usage = '%(prog)s'
												)
		
		self.parser.add_argument('-video', action='store_true')	
		self.parser.add_argument('-all', action='store_true')
		self.parser.add_argument('-load', action='store_true')
		self.parser.add_argument('-save', action='store_true')
		self.parser.add_argument('-view', action='store_true')

		if inOpts is None :
			self.args = self.parser.parse_args()
		else :
			self.args = self.parser.parse_args(inOpts)

def main(inCL = None):
	if inCL is None:
		cl = CommandLine()
	else :
		cl = CommandLine(inCL)

	img0 = 'frame0.jpg'
	img1 = 'frame1.jpg'
	window = 10
	dof = 0
	C = 3	
	write_num = 1	

	# still necessary?
	offset = 0
	offsetMult = 1
	adjust = -25
	point = (240, 260)

	if(cl.args.video):
		print ('displaying video')
		cap0 = cv2.VideoCapture(0)
		cap1 = cv2.VideoCapture(1)
		
		ret0, frame0 = cap0.read()
		assert ret0 # succeeds
		ret1, frame1 = cap1.read()
		assert ret1 # fails?!	

		while(True):	
			ret0, frame0 = cap0.read()
			ret1, frame1 = cap1.read()
			gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
			gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
			
			gray0 = cv2.adaptiveThreshold(gray0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,C)
			gray1 = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,C)
			
			if (cl.args.all):
				matchDict = matchFrames(gray0, gray1, window, 0)
				for key in matchDict:
					point = tup(key)
					match = tup(matchDict[key])
					gray0[point[ROW]:point[ROW]+window, point[COL]:point[COL]+window] = (point[ROW] + point[COL]) * 255 / 1120
					gray1[match[ROW]:match[ROW]+window, match[COL]:match[COL]+window] = (match[ROW] + match[COL]) * 255 / 1120	
	
			elif(cl.args.load):
				print('point')
				print(gray0[point[0]:point[0]+window,point[1]:point[1]+window].astype(int)-gray0[point[0]][point[1]].astype(int))
				print('line')
				offsetArr = gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window].astype(int)*offsetMult+offset
				print(offsetArr-offsetArr[0][0])
				gray0[point[0]:point[0]+window,point[1]:point[1]+window] = 120 # (point[0] + point[1]) * 255 / 1120
				gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window] = 120 # (point[0] + col) * 255 / 1120

			else:
				print('single point')
				match = matchPixels(gray0, gray1, point, window, dof)
				frame0[point[ROW]:point[ROW]+window, point[COL]:point[COL]+window] = (0, 0, 255) # (point[ROW] + point[COL]) * 255 / 1120
				frame1[match[ROW]:match[ROW]+window, match[COL]:match[COL]+window] = (0, 0, 255) # (match[ROW] + match[COL]) * 255 / 1120

			cv2.imshow('pointGray', gray0)
			cv2.imshow('lineGray', gray1)
			cv2.imshow('point', frame0)
			cv2.imshow('line', frame1)	
			# when everything done, release the capture
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break	
		
	else:
		print('reading from saved images')
		print('loading', img0, 'and', img1)
		frame0 = cv2.imread(img0)
		frame1 = cv2.imread(img1)
		gray0 = cv2.imread(img0, cv2.IMREAD_GRAYSCALE)
		gray1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
		gray0 = cv2.adaptiveThreshold(gray0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C)
		gray1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C)
	
		print('starting matching')
		matchDict = matchFrames(gray0, gray1, window, 0)
	
		for key in matchDict:
			point = tup(key)
			match = tup(matchDict[key])
			dist = int(realDistance(point[ROW], point[COL], match[ROW], match[COL]))
			
			if(dist > 0):
				frame0[point[ROW]][point[COL]] = (dist*255/300, 0, 0)
			else:
				frame0[point[ROW]][point[COL]] = (255, 255, 255)
		
		if (cl.args.save):
			cv2.imwrite(str(write_num)+'distanceframe'+str(C)+'w'+str(window)+'.jpg',frame0)
			# cv2.imwrite(str(write_num)+'matchframe'+str(C)+'w'+str(window)+'.jpg',frame1)
			# cv2.imwrite(str(write_num)+'distancegray'+str(C)+'w'+str(window)+'.jpg',gray1)
			# cv2.imshow('point', gray0)
		if (cl.args.view):
			cv2.imshow('pointGray', gray0)
			cv2.imshow('lineGray', gray1)
			cv2.imshow('point', frame0)
			cv2.imshow('line', frame1)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		with open(str(write_num) + img0 + '_' + img1 + '_' + 'w' + str(window) + '.json', 'w+') as f:
			json.dump(matchDict, f, indent=4)
			
if __name__ == '__main__':
	main()