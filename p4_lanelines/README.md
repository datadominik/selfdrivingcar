## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/image_1.png "Undistorted"
[image2]: ./examples/image_2.png "Sobel"
[image3]: ./examples/image_3.png "HLS filter"
[image4]: ./examples/image_4.png "Combined filter"
[image5]: ./examples/image_5.png "Warp source points"
[image6]: ./examples/image_6.png "Warped"
[image7]: ./examples/image_7.png "Binarized"
[image8]: ./examples/image_8.png "Histogram"
[image9]: ./examples/image_9.png "Lanes"
[image10]: ./examples/image_10.png "Output"
[image11]: ./examples/image_11.png "Output"
[video1]: ./final_submission.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./submission.ipynb" .  

I prepared a tensor of objects points which represents chessboard coners. For each calibration image the same objects points are jused. By applying `cv2.findChessboardCorners(gray, (9,6),None)` to every grayscaled calibration image, the real chessboard points are found. Then `cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)` is used to determine the matrix and distortion factor used to later undistort the images. For undistortion I use the cv2 function `cv2.undistort`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The lense distortion in this example seems to be rather low. By looking close at the edges of the checkerboard you can anyway recognize the correction effect.

**--- image: distortion correction**

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create the binary images used for further analysis I created a combination of Sobel and HLS transform as mentioned in the lecture. The code for Sobel transform is located in cell 2 of the jupyter notebook, the HLS code in cell 3. For both Sobel and HLS transform (where I focussed on the saturation channel) thresholds were chosen manually. Afterwards both filtered binary images were transformed through averaging.

**--- image: sobel filter**

![alt text][image2]

**--- image: hls filter**

![alt text][image3]

**--- image: combined filter**

![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is located in cells 5 and 6 of the jupyter notebook. Whereas the definition of source points was pretty straightforward, the definition of destination points was a big challenge in the first place. I therefore conducted the lesson videos again and browsed through the threads of the Udacity forum. There I could get a glimpse on how to tune the destination points - though, there was still lots of trial and error involved.

In the end I used the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 320, 720      |
| 450, 550      | 320, 576      |
| 840, 550      | 960, 576      |
| 1100, 720     | 960, 720      |

**--- image: source points**

![alt text][image5]

I then visually inspected the perspective transform by applying it to a test image and checking if the transform looks more or less valid.

**--- image: perspective transform**
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for lane finding is located in cells 7-11. It's heavily inspired by the examples and code of the sliding windows approach shown in the lecture. Though for me it worked better when I recomputed the histogram for every binarized image and used an interpolation between frames instead to ensure that the detection won't shake around too much.

**--- image: perspective transform on binary image**

![alt text][image7]

**--- image: histogram of first sliding window**

![alt text][image8]

**--- image: detected lanes**

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 13 of my jupyter notebook. For calculating the radius of curvature I reimplemented the code of the lecture. First, the detected lane pixels got transormed into meter space. After using the values in np.polyfit, I could then calculate the radius of curvature in real world space.

For detecting the offset to center, I first calculated the midpoint between both lines at the bottom of the image and substracted this value from the  midpoint of the image. Since we are still in pixel space, this value also got transformed into meter space afterwards.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 7-11 as well, since I evaluated the methodology on test images first. Here you see the result on a test image:

**--- image: result on example image**

![alt text][image10]

**--- image: result on video frame**

![alt text][image11]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Of course, one of the main problems here are all the manually tuned parameters. In fact, we're implicitly overfitting on the test videos, since the tuning is aimed at performing the detection on them. If we need to deal with previously unseen scenarios like different lightning conditions or not so clear lane colors, the algorithm is likely to fail.

To make the methods more robust I would suggest going away from the manual parameter tuning towards a more Machine Learning based approach. By using hand-labeled images as a target, the tuning could be automated and performed with a much larger amount of images in different lane scenarios. We could also step away from the traditional computer vision approaches and try out Neural Networks like U-Net to perform image segmentation based on previously hand-labeled images.
