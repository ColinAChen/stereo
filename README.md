# stereo
Stereo camera test

## Motivation
Extracting depth from images is a useful problem. Depth can assist in perception, three-dimensional reconstruction, and overall scene understanding. After my 2019 summer working with deep learning-based computer vision, as well as some structure-from-motion three dimensional reconstruction, I wanted to explore other more geometric based approaches to computer vision. I was excited to revisit and apply math like linear algebra and three-dimensional calculus that I had previously been unable to apply. I was also excited to revisit OpenCV and image processing.

## Techniques

#### Setup
We used two Logitech webcams. The specific models are unkown and this presents a few challenges.

We used [OpenCV](https://opencv.org/) for handling images and creating [adaptive thresholds](https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html).

We used [https://numpy.org/](Numpy) for matrix operations.

We referred to [https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect16.pdf](slides) from the University of Washington for learning about stereo camera calibration, epipolar geometry, and basic feature matching.

#### Pipline
We assume our cameras are pointed in the same direction at relatively the same angle. 

We perform feature matching by calculating the Sum of Squares Difference (SSD) between two windows in an image. We use absolute value instead of squaring which does not affect the determination for a best match as all values are above zero.
## For the Future
