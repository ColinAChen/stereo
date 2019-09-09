import cv2
import numpy as np
import threading
import time

# for working with tuples of the form (row, col)
ROW = 0
COL = 1

INF = 1_000_000_000	# represents infinity
DOF = 2			# number of pixels to search above or below corresponding row when matching pixels

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

def main():
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
	matchFrames(thresh0, thresh1, 100, 200, 100, 200, 22, window, DOF, 2)
	if(len(threads) > 0):
		threads[len(threads)-1].join()
	end = time.perf_counter()
	print("matching time:", end-start)

	color = (0, 0, 5)
	for key in matchDict:
		point0 = tup(key)
		point1 = tup(matchDict[key])
		box0 = frame0[point0[ROW]-int(window/2):point0[ROW]+int(window/2)+1, point0[COL]-int(window/2):point0[COL]+int(window/2)+1]
		box1 = frame1[point1[ROW]-int(window/2):point1[ROW]+int(window/2)+1, point1[COL]-int(window/2):point1[COL]+int(window/2)+1]
		for row in box0:
			for col in range(row.shape[0]):
				row[col] = color
		for row in box1:
			for col in range(row.shape[0]):
				row[col] = color

		if(color[2] != 255 and color[1] == 0 and color[0] == 0):
			color = (color[0], color[1], color[2]+5)
		elif(color[1] != 255 and color[2] == 255 and color[0] == 0):
			color = (color[0], color[1]+5, color[2])
		else:
			color = (color[0]+5, color[1], color[2])

	#cv2.imshow("gray0", gray0)
	#cv2.imshow("gray1", gray1)
	cv2.imshow("thresh0", thresh0)
	cv2.imshow("thresh1", thresh1)
	cv2.imshow("frame0", frame0)
	cv2.imshow("frame1", frame1)

	cv2.imwrite("frame0_thread_test2.jpg", frame0)
	cv2.imwrite("frame1_thread_test2.jpg", frame1)

	cv2.waitKey(0)

if __name__ == '__main__':
	main()
