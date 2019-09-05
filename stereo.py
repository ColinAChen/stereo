import cv2
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
	realP1 = (-10, -10)
	realP2 = (-10, -10)
	
	# check left wall for point
	testY = solveX(point[0], point[1], dir[0], dir[1], 0 - focalX)
	if(testY > 0 - focalY and testY < focalY):
		realP1[0] = 0 - focalX
		realP1[1] = testY
	# check bottom wall for point
	testX = solveY(point[0], point[1], dir[0], dir[1], 0 - focalY)
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
	testX = solveY(point[0], point[1], dir[0], dir[1], focalY)
	if(testX > 0 - focalX and testX < focalX):
		realP2[0] = testX
		realP2[1] = focalY

	return (realToPixel(realP1), realToPixel(realP2))
		
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
	while(True):
		# Capture frame-by-frame
		#ret, frame = cap.read()
		ret0, frame0 = cap0.read()
		ret1, frame1 = cap1.read()
		# Our operations on the frame come here
		#gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
		#gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

		#ret0,thresh0 = cv2.threshold(gray0,250,255,cv2.THRESH_BINARY)
		#ret1,thresh1 = cv2.threshold(gray1,250,255,cv2.THRESH_BINARY)
		# Display the resulting frame
		cv2.imshow('frame',frame0)
		cv2.imshow('frame1',frame1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# When everything done, release the capture
	cap0.release()
	cap1.release()
	cv2.destroyAllWindows()
if __name__ == '__main__':
	main()