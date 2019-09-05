import cv2
import matplotlib.pyplot as plt
from skimage.draw import line
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
	while(True):
		# Capture frame-by-frame
		#ret, frame = cap.read()
		ret0, frame0 = cap0.read()
		ret1, frame1 = cap1.read()
		# Our operations on the frame come here
		gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
		gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		for row in gray0:
			for col in row:
				matchX, matchY = match(row,col)




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