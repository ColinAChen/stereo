import cv2
import numpy as np
import multiprocessing
import time

import math
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

# stores matching points between frame0 and frame1
matchDict = {}
# stores subprocesses for multithreading/multiprocessing thingy
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
		elif(window > 80):
			break
		else:
			window += WIN_INC

	pointDict = {}	
	# iterate through search space in frame1
	for row in range(-1*dof, dof+1):
		row = row + point[ROW]
		# ignores pixels within half a window size from left and right sides
		for col in range(halfWin+1, frame1.shape[1]-halfWin):
			box1 = frame1[row+topRow:row+botRow, col+lefCol:col+rhtCol]
			boxDiff = sum2D(abs(np.subtract(box0.astype(int), box1.astype(int)))) / 255
			
			# store point values
			pointDict[str((row, col))] = boxDiff			
	
			if(boxDiff < minDiff):
				minDiff = boxDiff
				match = (row, col)

	#print("window: ", window, ", initial minDiff: ", minDiff, sep="")

	# search point values for best match
	matchThresh = 40
	for key in pointDict:
		if(pointDict[key] < matchThresh):
			if(abs(tup(key)[COL]-point[COL]) < abs(match[COL]-point[COL])):
				match = tup(key)
	return match
	

def matchFrames(frame0, frame1, rowStart, rowEnd, colStart, colEnd, inc, window, dof, numThreads=0, threadID=None, Q=None):
	localDict = {}
	numPixels = 0
	
	MAX_THREADS = frame0.shape[0]*frame0.shape[1]
	if(numThreads > 0 and numThreads < frame0.shape[0]):
		outputQ = []
		numRows = int((rowEnd-rowStart)/inc)
		rpt = int(numRows/numThreads)
		for i in range(numThreads):
			rS = rowStart+(i*rpt*inc)
			rE = rowStart+((i+1)*rpt*inc) if (i != numThreads-1) else rowEnd
			
			# was having issues overriding default value by name
			# was also having issues with queues filling up and subprocesses coming to a deadlock
			outputQ.append(multiprocessing.Queue())
			thread_args = (frame0, frame1, rS, rE, colStart, colEnd, inc, window, dof, 0, i, outputQ[i])
			threads.append(multiprocessing.Process(target=matchFrames, args=thread_args))

			print("Starting subprocess:", threads[i].name)
			threads[i].start() 

		for thread in threads:
			thread.join()
		for q in outputQ:
			while(not q.empty()):
				dict = q.get()
				for key in dict:
					matchDict[key] = dict[key]
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
	
	if(Q == None):
		for key in localDict:
			matchDict[key] = localDict[key]
	else:
		Q.put(localDict)

def saveDict(fileName):
	fp = open(fileName, "a")
	for key in matchDict:
		line = key + " " + matchDict[key] + "\n"
		fp.write(line)
	fp.close()

def loadDict(fileName):
	print("Loading from file", fileName)
	global matchDict
	matchDict = {}
	fp = open(fileName, "r")
	line = fp.readline()
	lineCount = 0
	while(line != ""):
		sep = 0
		while(line[sep] != ")"):
			sep += 1
		key = line[0:sep+1]
		val = line[sep+2:len(line)-1]
		matchDict[key] = val
	
		line = fp.readline()
		lineCount += 1
		if(lineCount % 1000 == 0):
			print("Completed line", lineCount)

def test2():
	window = 11
	frame0 = cv2.imread("frame0.jpg")
	frame1 = cv2.imread("frame1.jpg")
	frameDist = cv2.imread("frame0.jpg")
	loadDict("dict_0.05R_40CT.txt")
	
	maxDist = 0
	color = (0, 0, 0)
	for key in sorted(matchDict):
		point0 = tup(key)
		point1 = tup(matchDict[key])
		dist = realDistance(point0[ROW], point0[COL], point1[ROW], point1[COL])
 		
		print(dist)
		if(dist > maxDist):
			maxDist = dist

		halfWin = int(window/2)
		IMAGE_HEIGHT = frame0.shape[0]
		IMAGE_WIDTH = frame0.shape[1]
		topRow = -1*halfWin if (point0[ROW]-halfWin > 0) else -1*point0[ROW]
		botRow = halfWin+1 if (point0[ROW]+halfWin+1 < IMAGE_HEIGHT) else IMAGE_HEIGHT-point0[ROW]
		lefCol = -1*halfWin if (point0[COL]-halfWin > 0) else -1*point0[COL]
		rhtCol = halfWin+1 if (point0[COL]+halfWin+1 < IMAGE_WIDTH) else IMAGE_WIDTH-point0[COL]

		box0 = frame0[point0[ROW]+topRow:point0[ROW]+botRow, point0[COL]+lefCol:point0[COL]+rhtCol]
		box1 = frame1[point1[ROW]+topRow:point1[ROW]+botRow, point1[COL]+lefCol:point1[COL]+rhtCol]
		boxDist = frameDist[point0[ROW]+topRow:point0[ROW]+botRow, point0[COL]+lefCol:point0[COL]+rhtCol]
		for row in box0:
			for col in range(row.shape[0]):
				row[col] = color
		for row in box1:
			for col in range(row.shape[0]):
				row[col] = color
		for row in boxDist:
			for col in range(row.shape[0]):
				row[col] = (0, 0, 500/dist*255)
		
		if(color[0] == 0 and color[1] == 0 and color[2] != 255):
			color = (color[0], color[1], color[2]+0.5)
		elif(color[0] == 0 and color[1] != 255 and color[2] == 255):
			color = (color[0], color[1]+0.5, color[2])
		elif(color[0] == 0 and color[1] == 255 and color[2] != 0):
			color = (color[0], color[1], color[2]-0.5)
		elif(color[0] != 255 and color[1] == 255 and color[2] == 0):
			color = (color[0]+0.5, color[1], color[2])
		elif(color[0] == 255 and color[1] != 0 and color[2] == 0):
			color = (color[0], color[1]-0.5, color[2])
		elif(color[0] == 255 and color[1] == 0 and color[2] != 255):
			color = (color[0], color[1], color[2]+0.5)
		else:
			color = (color[0], color[1]+0.1, color[2])
	print("maxDist:", maxDist)
	
	cv2.imshow("frame0", frame0)
	cv2.imshow("frame1", frame1)
	cv2.imshow("frameDist", frameDist)
	cv2.waitKey(0)

def test1(startRow, endRow, startCol, endCol):
	frame0 = cv2.imread("frame0.jpg")
	frame1 = cv2.imread("frame1.jpg")
	gray0 = cv2.imread("frame0.jpg", cv2.IMREAD_GRAYSCALE)
	gray1 = cv2.imread("frame1.jpg", cv2.IMREAD_GRAYSCALE)
	
	C = 4
	MAX_VAL = 255
	window = 11
	thresh0 = cv2.adaptiveThreshold(gray0, MAX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, C)
	thresh1 = cv2.adaptiveThreshold(gray1, MAX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, C)
	
	# time multithreaded operation
	start = time.perf_counter()
	matchFrames(thresh0, thresh1, startRow, endRow, startCol, endCol, 11, window, DOF, 2)
	if(len(threads) > 0):
		threads[len(threads)-1].join()
	end = time.perf_counter()
	print("matching time:", end-start)

	'''
	color = (0, 0, 0)
	for key in sorted(matchDict):
		point0 = tup(key)
		point1 = tup(matchDict[key])

		halfWin = int(window/2)
		IMAGE_HEIGHT = frame0.shape[0]
		IMAGE_WIDTH = frame0.shape[1]
		topRow = -1*halfWin if (point0[ROW]-halfWin > 0) else -1*point0[ROW]
		botRow = halfWin+1 if (point0[ROW]+halfWin+1 < IMAGE_HEIGHT) else IMAGE_HEIGHT-point0[ROW]
		lefCol = -1*halfWin if (point0[COL]-halfWin > 0) else -1*point0[COL]
		rhtCol = halfWin+1 if (point0[COL]+halfWin+1 < IMAGE_WIDTH) else IMAGE_WIDTH-point0[COL]

		box0 = frame0[point0[ROW]+topRow:point0[ROW]+botRow, point0[COL]+lefCol:point0[COL]+rhtCol]
		box1 = frame1[point1[ROW]+topRow:point1[ROW]+botRow, point1[COL]+lefCol:point1[COL]+rhtCol]
		for row in box0:
			for col in range(row.shape[0]):
				row[col] = color
		for row in box1:
			for col in range(row.shape[0]):
				row[col] = color

		if(color[0] == 0 and color[1] == 0 and color[2] != 255):
			color = (color[0], color[1], color[2]+5)
		elif(color[0] == 0 and color[1] != 255 and color[2] == 255):
			color = (color[0], color[1]+5, color[2])
		elif(color[0] == 0 and color[1] == 255 and color[2] != 0):
			color = (color[0], color[1], color[2]-5)
		elif(color[0] != 255 and color[1] == 255 and color[2] == 0):
			color = (color[0]+5, color[1], color[2])
		elif(color[0] == 255 and color[1] != 0 and color[2] == 0):
			color = (color[0], color[1]-5, color[2])
		elif(color[0] == 255 and color[1] == 0 and color[2] != 255):
			color = (color[0], color[1], color[2]+5)
		else:
			color = (color[0], color[1]+5, color[2])
	
	#cv2.imshow("gray0", gray0)
	#cv2.imshow("gray1", gray1)
	cv2.imshow("thresh0", thresh0)
	cv2.imshow("thresh1", thresh1)
	cv2.imshow("frame0", frame0)
	cv2.imshow("frame1", frame1)
	
	cv2.imwrite("frame0_test.jpg", frame0)
	cv2.imwrite("frame1_test.jpg", frame1)

	cv2.waitKey(0)
	'''	

	saveDict("dict_0.05R_40CT.txt")

def main():
	'''
	for r in range(0, 481, 240):
		for c in range(0, 641, 120):
			sR = r
			eR = r + 240
			sC = c
			eC = c + 120 if (c + 120 < 640) else 640
			test1(sR, eR, sC, eC)
			print("finished sR:", sR, "eR:", eR, "sC:", sC, "eC:", eC)
			global matchDict
			matchDict = {}
			global threads
			threads = []
	'''
	test2()

if __name__ == '__main__':
	main()
