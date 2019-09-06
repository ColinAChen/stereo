import cv2
import stereo
from skimage.draw import line


image0 = cv2.imread("frame0.jpg")
image1 = cv2.imread("frame1.jpg")

point = [330, 455]
bounds = stereo.getSearchLine(point[0], point[1])
print("Point:", point[0], point[1])
print("Bounds:", bounds[0], bounds[1])

image0[point[0]][point[1]] = (0, 0, 255)
image1[330][432] = (0, 0, 255)

print(stereo.realDistance(330, 455,  330, 432))

if(False):
	cv2.imshow("window0", image0)
	cv2.imshow("window1", image1)
	cv2.waitKey(0)
