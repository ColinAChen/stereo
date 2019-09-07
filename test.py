import cv2
import stereo
from skimage.draw import line


image0 = cv2.imread("frame0.jpg")
image1 = cv2.imread("frame1.jpg")

point = [200, 0]
point2 = [200,400]
bounds = stereo.getSearchLine(point[0], point[1])
print("Point:", point[0], point[1])
print("Bounds:", bounds[0], bounds[1])

image0[point[0]][point[1]] = (0, 0, 255)
image1[point2[0]][point2[1]] = (0, 0, 255)

print(stereo.realDistance(point[0], point[1],  point2[0], point2[1]))

if(True):
	cv2.imshow("window0", image0)
	cv2.imshow("window1", image1)
	cv2.waitKey(0)
