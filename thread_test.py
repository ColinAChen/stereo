import cv2
import numpy as np

# for working with tuples of the form (row, col)
ROW = 0
COL = 1

INF = 1_000_000_000	# represents infinity
DOF = 2			# number of pixels to search above or below corresponding row when matching pixels

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
	WIN_DEFAULT = 11
	WIN_INC = 4
	WIN_MIN_RATIO = 0.05
	WIN_MIN_NUM_BW = INF
	minDiff = INF
	match = (-1, -1)

	# increase window size until ratio of black to white pixels is acceptable
	box0 = -1
	while(True):
		halfWin = int(window/2)
		box0 = frame0[point[ROW]-halfWin:point[ROW]+halfWin+1, point[COL]-halfWin:point[COL]+halfWin+1]
		
		black, white = countBW(box0)
		if(black == 0 or white == 0):			# all black or white
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
			box1 = frame1[row-halfWin:row+halfWin+1, col-halfWin:col+halfWin+1]
			boxDiff = sum2D(abs(np.subtract(box0.astype(int), box1.astype(int)))) / 255
			if(boxDiff < minDiff):
				minDiff = boxDiff
				match = (row, col)

	print("window size: ", window, ", minDiff: ", minDiff, sep="")
	return match

def matchFrames(frame0, frame1, window, dof):
	matchDict = {}
	numPixels = 0

	# iterate through the frame
	#for row in range(window/2+1, frame0.shape[0]-window/2):
	for row in range(100, 321, 22):
		#for col in range(window/2+1, frame0.shape[1]-window/2):
		for col in range(100, 321, 22):
			point = (row, col)
			match = matchPixels(frame0, frame1, (row, col), window, dof)
			matchDict[str(point)] = str(match)
			numPixels += 1

			if(numPixels % 20 == 0):
				print("Pixels completed:", numPixels)

	return matchDict		

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
	
	matches = matchFrames(thresh0, thresh1, window, DOF)
	color = (0, 0, 5)
	for key in matches:
		point0 = tup(key)
		point1 = tup(matches[key])
		box0 = frame0[point0[ROW]-int(window/2):point0[ROW]+int(window/2)+1, point0[COL]-int(window/2):point0[COL]+int(window/2)+1]
		box1 = frame1[point1[ROW]-int(window/2):point1[ROW]+int(window/2)+1, point1[COL]-int(window/2):point1[COL]+int(window/2)+1]
		for row in box0:
			for col in range(row.shape[0]):
				row[col] = color
		for row in box1:
			for col in range(row.shape[0]):
				row[col] = color

		if(color[2] != 255):
			color = (color[0], color[1], color[2]+5)
		elif(color[1] != 255):
			color = (color[0], color[1]+5, color[2])
		else:
			color = (color[0]+5, color[1], color[2])

	#cv2.imshow("gray0", gray0)
	#cv2.imshow("gray1", gray1)
	cv2.imshow("thresh0", thresh0)
	cv2.imshow("thresh1", thresh1)
	cv2.imshow("frame0", frame0)
	cv2.imshow("frame1", frame1)

	cv2.imwrite("frame0_0.05RATIO_OVER50.jpg", frame0)
	cv2.imwrite("frame1_0.05RATIO_OVER50.jpg", frame1)

	cv2.waitKey(0)

if __name__ == '__main__':
	main()
