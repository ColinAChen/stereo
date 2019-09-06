import cv2
import matplotlib.pyplot as plt
from skimage.draw import line
import numpy as np
import PIL
import math
#Real world cm height/width
realWidth = .86808
realHeight = .65106
#focal distance
focalDistance = 1.915
focalX = realWidth/2
focalY = realWidth/2
#pixel dimensions:
pixelWidth = 640
pixelHeight = 480

def pixelToReal(x,y):
	'''
	return: 3D coordinates in cm of a point on the image plane
	'''
	cmx = realWidth * (x/pixelWidth)
	cmy = realHeight * (y/pixelHeight)
	return (focalDistance,cmx-focalX,cmy-focalY)
def realToPixel(cmx,cmy):
	'''
	return: 2D cordinates of a pixel from 3D coordinates in cm
	'''
	x = (cmx + focalX)*pixelWidth/realWidth
	y = (cmy + focalY)*pixelHeight/realHeight
	return (x,y)


def match(img1,x,y,img2,linex1,liney1,linex2,liney2):
	diff = 255
	matchRow = None
	matchCol = None
	rr,cc = line(linex1,liney1,linex2,liney2)
	for row in rr,cc:
		for col in row:
			if(abs(img1[x][y] - img2[row][col]) < diff):
				diff = abs(img1[x][y] - img2[row][col]) < diff
				matchRow = row
				matchCol = col
	return matchRow,matchCol



#def realDistance()

def getSearchLine(pixelX, pixelY):
	firstFP = (0, 0, 0)
	secondFP = (0, 9, 0)	
		
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

	return (realToPixel(realP1[0],realP1[1]), realToPixel(realP2[0],realP2[1]))
		
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
System of the form:
Ax + By = E
Cx + Dy = F
returns solution (x, y) as tuple
'''
def solveSystem(A, B, C, D, E, F):	
	inverseCoefficient = (1 / (A*D - B*C))
	xSol = inverseCoefficient * (A*E - B*F)
	ySol = inverseCoefficient * (D*F - C*E)
	return(xSol, ySol)

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

def main():
	cap0 = cv2.VideoCapture(0)
	cap1 = cv2.VideoCapture(1)
	ret0, frame0 = cap0.read()
	assert ret0 # succeeds
	ret1, frame1 = cap1.read()
	assert ret1 # fails?!
	#cap0.set(3,1920)
	#cap0.set(4,1080)
	#print(cap0.get(3))
	#print(cap0.get(4))
	#stereo = cv2.StereoBM_create(0,21)
	thresh = 10
	point=[240,260]
	saved = False
	while(True):
		if (not saved):
			cv2.imwrite('frame0.jpg',frame0)
			cv2.imwrite('frame1.jpg',frame1)
			saved=True
		# Capture frame-by-frame
		#ret, frame = cap.read()
		ret0, frame0 = cap0.read()
		ret1, frame1 = cap1.read()
		# Our operations on the frame come here
		gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
		gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		#bounds = getSearchLine(point[0],point[1])
		#print(bounds)
		#frame0[point[0]][point[1]] = (255,255,255)
		#rr,cc = line(round(bounds[0][0]),round(bounds[0][1]),round(bounds[1][0]-1),round(bounds[1][1]))
		minDist = 255
		minRow = -10
		minCol = -10
		'''
		for i,row in enumerate(rr):
			#print(row,cc[i])
			#frame1[cc[i]][row] = (255,0,0)

			#print(dist(frame0[point[0]][point[1]][0], frame1[cc[i]][row][0],frame0[point[0]][point[1]][1], frame1[cc[i]][row][1],frame0[point[0]][point[1]][2], frame1[cc[i]][row][2]))
			if (dist(frame0[point[0]][point[1]][0], frame1[cc[i]][row][0],frame0[point[0]][point[1]][1], frame1[cc[i]][row][1],frame0[point[0]][point[1]][2], frame1[cc[i]][row][2]) < minDist):
				#print('dist match found!')
				#print(dist(frame0[point[0]][point[1]][0], frame1[cc[i]][row][0],frame0[point[0]][point[1]][1], frame1[cc[i]][row][1],frame0[point[0]][point[1]][2], frame1[cc[i]][row][2]))
				minDist = dist(frame0[point[0]][point[1]][0], frame1[cc[i]][row][0],frame0[point[0]][point[1]][1], frame1[cc[i]][row][1],frame0[point[0]][point[1]][2], frame1[cc[i]][row][2])
				minRow = row
				minCol = cc[i]
				
				#print(cc[i])
				#print([row])
		print(minRow)
		print(minCol)
		frame1[minCol][minRow] = (255,255,255)
		frame0[point[0]][point[1]] = (255,255,255)
		'''
		#print(rr)
			#if (abs(frame0[point[0]][point[1]][0] - frame1[cc[i]][row][0]) < thresh and abs(frame0[point[0]][point[1]][1] - frame1[cc[i]][row][1]) < thresh and abs(frame0[point[0]][point[1]][2] - frame1[cc[i]][row][2]) < thresh):
			#	print('match found!')
			#	frame1[cc[i]][row] = (255,255,255)
		#print(rr)
		#print(cc)
		#show = np.asarray(gray1)
		#print(show)
		#print(len(show))
		#show[rr][cc] = 255
		#gray1[rr][cc] = 255

		#for row in range(0,480,4):
		#	minPoint = 1000
		#	minCol = -10
		#	for col in range(0,640,4):
		#		if (areaDist(frame0[row:row+4][col:col+4],frame1[[row:row+4][col:col+4]]) < minPoint):
		#			minPoint = areaDist(frame0[row:row+4][col:col+4],frame1[[row:row+4][col:col+4]])
		#			minCol = col
		#	frame1[]
		#if (areaDist(frame0[point[0]:point[0]+20,point[1]:point[1]+20],frame1[point[0]:point[0]+20,col:col+20]) < minPoint):
		#	minPoint = areaDist(frame0[point[0]:point[0]+20,col:col+20],frame1[point[0]:point[0]+20,col:col+20])
		#	minCol = col
		for row in range(0,480,40):
			for col in range(0,640,40):
				minPoint = 10000000000
				minCol = -10
				#temp = 1000
				#print(sum(np.subtract(gray0[row:row+20,col:col+20],gray1[row:row+20,col:col+20])))
				for slideCol in range(0,620):
					if (sum(sum(abs(np.subtract(gray0[row:row+20,col:col+20],gray1[row:row+20,slideCol:slideCol+20])))) < minPoint):
						minPoint = sum(sum(abs(np.subtract(gray0[row:row+20,col:col+20],gray1[row:row+20,slideCol:slideCol+20]))))
						minCol = slideCol
				#print('column', col)
				#print('minPoint',minPoint)
				#print('minCol', minCol)
				gray0[row:row+20,col:col+20] = (row + col) * 255 / 1120
				gray1[row:row+20,minCol:minCol+20] = (row + col) * 255 / 1120


		#gray1[point[0]:point[0]+10,point[1]-12:point[1]+10-12] = 255
		cv2.imshow('point',gray0)
		cv2.imshow('line', gray1)

		


		#orb = cv2.ORB_create()

		#kp1, des1 = orb.detectAndCompute(gray0,None)
		#kp2, des2 = orb.detectAndCompute(gray1,None)

		# create BFMatcher object
		#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# Match descriptors.
		#matches = bf.match(des1,des2)

		# Sort them in the order of their distance.
		#matches = sorted(matches, key = lambda x:x.distance)
		#print(des1[matches[0].trainIdx])
		# Draw first 10 matches.
		#img3 = cv2.drawMatches(gray0,kp1,gray1,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		#cv2.imshow('matches',img3)
		#plt.imshow(img3),plt.show()
		#out=[]
		#disparity = stereo.compute(gray0,gray1)
		#retval = stereo.getMinDisparity()
		#print(retval)
		#print(disparity)
		#ret0,thresh0 = cv2.threshold(gray0,250,255,cv2.THRESH_BINARY)
		#ret1,thresh1 = cv2.threshold(gray1,250,255,cv2.THRESH_BINARY)
		# Display the resulting frame
		#cv2.imshow('frame',frame0)
		#cv2.imshow('frame1',frame1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# When everything done, release the capture
	cap0.release()
	cap1.release()
	cv2.destroyAllWindows()
if __name__ == '__main__':
	main()