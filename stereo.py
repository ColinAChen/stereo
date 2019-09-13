import cv2
import matplotlib.pyplot as plt
from skimage.draw import line
import numpy as np
import PIL
import math
import argparse
import json
import threading
import time

#Real world cm height/width
realWidth = .86808
realHeight = .65106
# distance between focal point and frame
focalDistance = 0.625 #1.915
focalX = realWidth/2
focalY = realHeight/2
# distance between focal points
frameDistance = 9
#pixel dimensions:
pixelWidth = 640
pixelHeight = 480

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
	
# input is pixel coordinates in frame1
# output is pixel coordinates in frame2
def getSearchLine(pixelX, pixelY):
		
	# get initial line	
	coord = pixelToReal(pixelX, pixelY)
	
	# get initial plane
	initDirec = crossProduct(coord, secondFP)
	initPoint = (0, 0, 0)

	# get frame2 plane
	frameDirec = (1, 0, 0)
	framePoint = (focalDistance, 9, 0)

	# get intersection of 2 planes
	# line will be: <point[0], point[1]> + t<dir[0], dir[1]>
	dir = crossProduct(initDirec, frameDirec)
	dir = (dir[1], dir[2])

	point = (0, (dotProduct(initPoint, initDirec)-initDirec[0]*focalDistance) / initDirec[2])

	
	# check for bounds of plane
	# uninitialized point values are -10
	realP1 = [-10, -10]
	realP2 = [-10, -10]
	
	# check left wall for point
	testY = solveX(point[0], point[1], dir[0], dir[1], 0 - focalX)
	if(testY > 0 - focalY and testY < focalY):
		realP1[0] = 0 - focalX
		realP1[1] = testY
	# check bottom wall for point
	if (dir[1] != 0):
		testX = solveY(point[0], point[1], dir[0], dir[1], 0 - focalY)
	else:
		testX = -10
	if(testX > 0 - focalX and testX < focalX):
		if(realP1[0] == -10):
			realP1[0] = testX
			realP1[1] = 0 - focalY
		else:
			realP2[0] = testX
			realP2[1] = 0 - focalY
	# check right wall for point
	testY = solveX(point[0], point[1], dir[0], dir[1], focalX)
	if(testY > 0 - focalY and testY < focalY):
		if(realP1[0] == -10):
			realP1[0] = focalX
			realP1[1] = testY
		else:
			realP2[0] = focalX
			realP2[1] = testY
	# check top wall for point
	if (dir[1] != 0):
		testX = solveY(point[0], point[1], dir[0], dir[1], focalY)
	else:
		testX = -10
	if(testX > 0 - focalX and testX < focalX):
		realP2[0] = testX
		realP2[1] = focalY

	return (realToPixelFrame2(realP1[0],realP1[1]), realToPixel(realP2[0],realP2[1]))
		
def crossProduct(vec1, vec2):
	x = vec1[1] * vec2[2] - vec1[2] * vec2[1]
	y = vec1[2] * vec2[0] - vec1[0] * vec2[2]
	z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
	return (x, y, z)

def dotProduct(vec1, vec2):
	x = vec1[0] * vec2[0]
	y = vec1[1] * vec2[1]
	z = vec1[2] * vec2[2]
	return x + y + z
'''
solveX solves for y
solveY solves for x
eq of form:
<P1, P2> + t<D1, D2> = <x, y>
'''
def solveX(P1, P2, D1, D2, x):
	t = (x - P1) / D1
	return P2 + t * D2

def solveY(P1, P2, D1, D2, y):
	t = (y - P2) / D2
	return P1 + t * D1

def dist(a,b,c,d,e,f):
	#print('dist')
	#print(a,b,c,d,e,f)
	return math.sqrt((int(a)-int(b))**2 + (int(c)-int(d))**2 + (int(e)-int(f))**2)

def areaDist(area1, area2):
	'''
	area1: list of lists or tuples
	area2: list of lists or tuples
	return: distance between
	'''
	distance=0
	for i, row in enumerate(area1):
		distance+=listDist(row,area2[i])
	return distance/len(area1)

def listDist(list1,list2):
	'''
	list1: list or tuple of colors with three channels
	list2: list of tupe or colors with three channels
	return: distance between list of colors
	'''

	distance = 0
	for i, point in enumerate(list1):
		distance += dist(point[0],list2[i][0],point[1],list2[i][1],point[2],list2[i][2])
	return distance/len(list1)

def diffArray(twoD):
	'''
	twoD: list of list
	return: difference array, top left corner's value is the same. All other values are their corresponding pixel's relationship to the to left anchor pixel.
	'''
	pass
	'''
	anchor = twoD[0][0]
	out = [][]
	for i,row in enumerate(twoD):
		for j,col in enumerate(row):
			out[i][j] = col-anchor
	return out
	'''


# for working with tuples of the form (row, col)
ROW = 0
COL = 1

INF = 1000000000	# represents infinity
# DOF = 2			# number of pixels to search above or below corresponding row when matching pixels

# stores matching points between frame0 and frame1
matchDict = {}
# stores threads for multithreading
threads = []

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

# slightly faster than using sum(sum(arr)) I think
def sum2D(arr):
	total = 0
	for row in arr:
		for elem in row:
			total += elem
	return total	

# counts number of black and white pixels in grayscale array
# returns in order numBlack, numWhite
def countBW(arr):
	BLACK = 0
	WHITE = 255

	numBlack = 0
	numWhite = 0
	for row in arr:
		for elem in row:
			if(elem == BLACK):
				numBlack += 1
			elif(elem == WHITE):
				numWhite += 1
	return numBlack, numWhite			

# point is a tuple representing (row, col) of point on frame0
# function searches for a matching point on frame1
# dof is the degrees of freedom allowed to search away from the corresponding row
def matchPixels(frame0, frame1, point, window, dof):
	IMAGE_HEIGHT = frame0.shape[0]
	IMAGE_WIDTH = frame0.shape[1]
	if(point[ROW] < 0 or point[ROW] > IMAGE_HEIGHT):
		return (-1, -1)
	if(point[COL] < 0 or point[COL] > IMAGE_WIDTH):
		return (-1, -1)

	WIN_DEFAULT = 11
	WIN_INC = 4
	WIN_MIN_RATIO = 0.05
	WIN_MIN_NUM_BW = INF

	minDiff = INF
	match = (-1, -1)

	# increase window size until ratio of black to white pixels is acceptable
	box0 = -1
	topRow = -1
	botRow = -1
	lefCol = -1
	rhtCol = -1
	while(True):
		halfWin = int(window/2)
		topRow = -1*halfWin if (point[ROW]-halfWin > 0) else -1*point[ROW]
		botRow = halfWin+1 if (point[ROW]+halfWin+1 < IMAGE_HEIGHT) else IMAGE_HEIGHT-point[ROW]
		lefCol = -1*halfWin if (point[COL]-halfWin > 0) else -1*point[COL]
		rhtCol = halfWin+1 if (point[COL]+halfWin+1 < IMAGE_WIDTH) else IMAGE_WIDTH-point[COL]
 
		box0 = frame0[point[ROW]+topRow:point[ROW]+botRow, point[COL]+lefCol:point[COL]+rhtCol]
		
		black, white = countBW(box0)
		if(black == 0 or white == 0):
			window += WIN_INC
		elif(black/(black+white) > WIN_MIN_RATIO and white/(black+white) > WIN_MIN_RATIO):
			break
		elif(black > WIN_MIN_NUM_BW and white > WIN_MIN_NUM_BW):
			break
		else:
			window += WIN_INC

	# iterate through search space in frame1
	for row in range(-1*dof, dof):
		row = row + point[ROW]
		# ignores pixels within half a window size from left and right sides
		for col in range(halfWin+1, frame1.shape[1]-halfWin):
			box1 = frame1[row+topRow:row+botRow, col+lefCol:col+rhtCol]
			boxDiff = sum2D(abs(np.subtract(box0.astype(int), box1.astype(int)))) / 255
			if(boxDiff < minDiff):
				minDiff = boxDiff
				match = (row, col)

	# print("window size: ", window, ", minDiff: ", minDiff, sep="")
	return match

def matchFrames(frame0, frame1, rowStart, rowEnd, colStart, colEnd, inc, window, dof, numThreads=0, threadID=None):
	if(threadID != None):
		print("starting thread", threads[threadID])
	localDict = {}
	numPixels = 0
	
	MAX_THREADS = frame0.shape[0]*frame0.shape[1]
	if(numThreads > 0 and numThreads < frame0.shape[0]):
		numRows = int((rowEnd-rowStart)/inc)
		rpt = int(numRows/numThreads)
		for i in range(numThreads):
			rS = rowStart+(i*rpt*inc)
			rE = rowStart+((i+1)*rpt*inc) if (i != numThreads-1) else rowEnd
			
			# was having issues overriding default value by name
			thread_args = (frame0, frame1, rS, rE, colStart, colEnd, inc, window, dof, 0, i)
			threads.append(threading.Thread(target=matchFrames, args=thread_args))
			threads[i].start() 
		return

	# iterate through the frame
	for row in range(rowStart, rowEnd, inc):
		for col in range(colStart, colEnd, inc):
			point = (row, col)
			match = matchPixels(frame0, frame1, (row, col), window, dof)
			localDict[str(point)] = str(match)
			numPixels += 1

			if(numPixels % 20 == 0):
				print("Pixels completed:", numPixels)	

	if(threadID != None and threadID != 0):
		threads[threadID-1].join()

	for key in localDict:
		matchDict[key] = localDict[key]

def matchFrames1():
	for row in range(window,frame0.shape[0]-window):
		print('Matching pixels on row',row)
		for col in range(window,frame0.shape[1]-window):
				minPoint = 10000000000
				minCol = -10
				minRow = -10
				#temp = 1000
				#print(sum(np.subtract(gray0[row:row+20,col:col+20],gray1[row:row+20,col:col+20])))
				for slideCol in range(0,frame0.shape[1]-window):
					for DOF in range(0-int(window/2),1+int(window/2)):
						if (sum(sum(abs(np.subtract(
											gray0[row+DOF:row+DOF+window,col:col+window],
											gray1[row+DOF:row+DOF+window,slideCol:slideCol+window])))) < minPoint):
							
							minPoint = sum(sum(abs(np.subtract(gray0[row+DOF:row+DOF+window,col:col+window],
											gray1[row+DOF:row+DOF+window,slideCol:slideCol+window]))))
							minCol = slideCol
							minRow = row+DOF
				matchDict[str((row,col))] = [minRow,minCol]
				if (int(realDistance(row,col,minRow,minCol)) > 0):
					frame0[row][col] = ((int(realDistance(row,col,minRow,minCol)) * 255 / 300),0,0)#int(realDistance(firstRow[i],firstCol,minRow,minCol)) * 255/120)
				else:
					frame0[row][col] = (255,255,255)
				#for r in range(0,window):
					#for c in range(0,window):
						#print('distance at',row+r,col+c,realDistance(row+r,col+c,minRow+r,minCol+c))
						#print(firstRow[i],firstCol)
						#print(realDistance(firstRow[i],firstCol,firstRow[i],minCol))
						#if (realDistance(firstRow[i],firstCol,firstRow[i],minCol) < 150):
						#gray0[firstRow[i]][firstCol] = int(realDistance(firstRow[i],firstCol,firstRow[i],minCol)) * 255/120
						
				'''
				for i,firstRow in enumerate(gray0[row:row+window]):
					for firstCol in firstRow[col:col+window]:
						print('distance at',firstRow[i],firstCol,realDistance(firstRow[i],firstCol,minRow,minCol))
						#print(firstRow[i],firstCol)
						#print(realDistance(firstRow[i],firstCol,firstRow[i],minCol))
						#if (realDistance(firstRow[i],firstCol,firstRow[i],minCol) < 150):
						#gray0[firstRow[i]][firstCol] = int(realDistance(firstRow[i],firstCol,firstRow[i],minCol)) * 255/120
						if (int(realDistance(firstRow[i],firstCol,minRow,minCol)) > 0):
							frame0[firstRow[i]][firstCol] = (255/int(realDistance(firstRow[i],firstCol,minRow,minCol)),0,0)#int(realDistance(firstRow[i],firstCol,minRow,minCol)) * 255/120)
						else:
							frame0[firstRow[i]][firstCol] = (0,0,0)
				'''
				#gray0[row:row+window,col:col+window] = (row + col) * 255 / 1120
				#gray1[row:row+window,minCol:minCol+window] = (row + col) * 255 / 1120
				#gray0[row:row+window,col:col+window] = (row + col) * 255 / 1120
				#gray1[minRow:minRow+window,minCol:minCol+window] = (minRow + minCol) * 255 / 1120
				#frame0[row:row+window,col:col+window] = (row*255/480,0,col*255/640)
				#frame1[minRow:minRow+window,minCol:minCol+window] = (minRow*255/480,0,minCol*255/640)
		#cv2.imshow('pointGray',gray0)
		#cv2.imshow('lineGray', gray1)
		#cv2.imshow('point',frame0)
		#cv2.imshow('line', frame1)
		#cv2.imwrite('frame0'+str(c) + 'G.jpg',frame0)

def main(inCL = None):
	if inCL is None:
		cl = CommandLine()
	else :
		cl = CommandLine(inCL)

	imgName0 = 'frame0.jpg'
	imgName1 = 'frame1.jpg'
	window = 10
	offset = 0
	offsetMult = 1
	adjust = -25
	C = 3
	write_num = 1
	matchDict = {}
	if (cl.args.video):
		print ('video')
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
			
			print(c%15)
			gray0 = cv2.adaptiveThreshold(gray0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,c%15)
			gray1 = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,c%15)
			
			minDist = 255
			minRow = -10
			minCol = -10
			
			if (cl.args.all):
				for row in range(0,480,window*2):
					for col in range(0,640,window*2):
						minPoint = 10000000000
						minCol = -10
						
						for slideCol in range(0,640-window):
							if (sum(sum(abs(np.subtract(gray0[point[0]:point[0]+window,point[1]:point[1]+window].astype(int)-gray0[point[0]][point[1]].astype(int),gray1[point[0]:point[0]+window,slideCol:slideCol+window].astype(int)-gray1[point[0]][minCol].astype(int) + offset)))) < minPoint):
								minPoint = sum(sum(abs(np.subtract(gray0[point[0]:point[0]+window,point[1]:
									point[1]+window].astype(int)-gray0[point[0]][point[1]].astype(int),
									gray1[point[0]:point[0]+window,slideCol:slideCol+window].astype(int)-gray1[point[0]][minCol].astype(int) + offset))))
								minCol = slideCol
					
						gray0[row:row+window,col:col+window] = (row + col) * 255 / 1120
						gray1[row:row+window,minCol:minCol+window] = (row + col) * 255 / 1120
			elif(cl.args.load):
				print('point')
				print(gray0[point[0]:point[0]+window,point[1]:point[1]+window].astype(int)-gray0[point[0]][point[1]].astype(int))
				print('line')
				offsetArr = gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window].astype(int) * offsetMult + offset
				#print(gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window].astype(int))
				#print(offsetArr)
				print(offsetArr-offsetArr[0][0])
				#print(gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window].astype(int) + offset - gray1[point[0]][point[1]-adjust].astype(int))
				gray0[point[0]:point[0]+window,point[1]:point[1]+window] = 120#(point[0] + point[1]) * 255 / 1120
				gray1[point[0]:point[0]+window,point[1]-adjust:point[1]-adjust+window] = 120#(point[0] + col) * 255 / 1120
				#gray1[point[0]:point[0]+10,point[1]-12:point[1]+10-12] = 255
			else:
				
				#print('single point')
				for col in range(0,640,window):
					minPoint = 10000000000
					minCol = -10
					#temp = 1000
					#print(sum(np.subtract(gray0[row:row+20,col:col+20],gray1[row:row+20,col:col+20])))
					for slideCol in range(0,640-window):
						for DOF in range(-3,4):

							if (point[0]+DOF >= 0 and point[0]+DOF <= 480 and sum(sum(abs(np.subtract(gray0[point[0]+DOF:point[0]+DOF+window,point[1]:point[1]+window].astype(int)-gray0[point[0]][point[1]].astype(int),gray1[point[0]:point[0]+window,slideCol:slideCol+window].astype(int) + offset - gray1[point[0]][minCol].astype(int))))) < minPoint):
								minPoint = sum(sum(abs(np.subtract(gray0[point[0]+DOF:point[0]+DOF+window,point[1]:
									point[1]+window].astype(int)-gray0[point[0]+DOF][point[1]].astype(int),
									gray1[point[0]+DOF:point[0]+DOF+window,slideCol:slideCol+window].astype(int) + offset - gray1[point[0]+DOF][minCol].astype(int)))))
								minCol = slideCol
					#print('column', col)
					#print('minPoint',minPoint)
					#print('minCol', minCol)
				#print('point')
				#print(gray0[point[0]:point[0]+window,point[1]:point[1]+window])
				#print('point')
				#print(gray0[point[0]:point[0]+window,point[1]:point[1]+window].astype(int) - gray0[point[0]][point[1]].astype(int))
				#print('line')
				#print(gray1[point[0]:point[0]+window,minCol:minCol+window].astype(int) + offset - gray1[point[0]][minCol].astype(int))
				#gray0[point[0]:point[0]+window,point[1]:point[1]+window] = 255#(point[0] + point[1]) * 255 / 1120
				#gray1[point[0]:point[0]+window,minCol:minCol+window] = 255#(point[0] + col) * 255 / 1120
				frame0[point[0]+DOF:point[0]+DOF+window,point[1]:point[1]+window] = 0,0,255#(point[0] + point[1]) * 255 / 1120
				frame1[point[0]+DOF:point[0]+DOF+window,minCol:minCol+window] = 0,0,255#(point[0] + col) * 255 / 1120
			cv2.imshow('pointGray',gray0)
			cv2.imshow('lineGray', gray1)
			cv2.imshow('point',frame0)
			cv2.imshow('line', frame1)
			#cv2.imshow('third',gray2)
			# When everything done, release the capture
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	else:
		THRESH_WIN = 11

		print('reading from saved images')
		print('loading', imgName0, 'and', imgName1)
		
		frame0 = cv2.imread(imgName0)
		frame1 = cv2.imread(imgName1)
		gray0 = cv2.imread(imgName0, cv2.IMREAD_GRAYSCALE)
		gray1 = cv2.imread(imgName1, cv2.IMREAD_GRAYSCALE)
		gray0 = cv2.adaptiveThreshold(gray0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, THRESH_WIN, C)
		gray1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, THRESH_WIN, C)
		
		print('starting matching')
		
		matchFrames1()

		if (cl.args.save):
			cv2.imwrite(str(write_num) + 'distanceframe'+str(c)+'w'+str(window)+'.jpg',frame0)
			#cv2.imwrite(str(write_num) + 'matchframe'+str(c)+'w'+str(window)+'.jpg',frame1)
			#cv2.imwrite(str(write_num) + 'distancegray'+str(c)+'w'+str(window)+'.jpg',gray1)
			#cv2.imshow('point',gray0)
		if (cl.args.view):
			cv2.imshow('pointGray',gray0)
			cv2.imshow('lineGray', gray1)
			cv2.imshow('point',frame0)
			cv2.imshow('line', frame1)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		with open(str(write_num) + imgName0 + '_' + imgName1 + '_' + 'w' + str(window) + '.json','w+') as f:
			json.dump(matchDict,f,indent=4)

if __name__ == '__main__':
	main()