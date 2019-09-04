import cv2
#Real world cm height/width
realWidth = .86808
realHeight = .65106
#focal distance
focalDistance = 1.915
focalX = realWidth/2
focalY = realWidth/2
#pixel dimensions:


def pixelToReal(x,y):
	'''
	return: 3D coordinates in cm of a point on the image plane
	'''
	cmx = realWidth * (x/640)
	cmy = realHeight * (y/480)
	return (1.915,cmx-focalX,cmy-focalY)
def realToPixel(cmx,cmy):
	'''
	return: 2D cordinates of a pixel from 3D coordinates in cm
	'''
	x = (cmx + focalX)*640/realWidth
	y = (cmy + focalY)*480/realHeight
	return (x,y)


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