#**Finding Lane Lines on the Road**

##Writeup
###Author: Dominik Schniertshauer
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./examples/transformed.jpg "Transformed images"
[video1]: ./examples/video1.JPG "Result video 1"
[video2]: ./examples/video2.JPG "Result video 2"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline I built consited of the following steps:
* read in the video frame and make a copy of it
* transfer 3-channel RGB image to grayscale
* apply gaussian blur filter to reduce level of detail and getting rid of noise
* apply canny filter to detect potential edges
* focus on a certain region of interest since we have a fixed camera position
* transform hough transform to transform our point space to lines
* detect which lines are candidates for left lane markers and which for right lane markers
* extrapolate the lines with linear regression to find a "one fits all" line

Our goal was to draw a single line for the left and right lane. To achieve this I first wrote a function that checks the slope of each hough transform line. If we are over a positive/ negative threshold each single line gets assigned to the suitable lane.

Since the hough transform produces too many lines I now construct one single line for each lane. Simple linear regression is used to find a function that describes all points of the hough lines with an error as little as possible. Since we always want to start the line at the bottom we use the formula **y = m*x + b** and transform it to **x = (y-b)/m**. Like this we get the needed x-position for each line if you are at the bottom of the image. I also restrict the end of the linear line to a certain y-position, since only parts of the real lane are describable with a linear model due to the viewpoint.

Here are some image examples with my lane detection pipeline:
![alt text][image2]

Here is a screenshot of video 1 with my lane detection pipeline:
![alt text][video1]

And here is a screenshot of video 2 with my lane detection pipeline:
![alt text][video2]

###2. Identify potential shortcomings with your current pipeline

From my point of view there are several shortcomings:
1. **hand-tuned parameters**: for edge detection, hough transform and region cropping I use parameters that have been tuned by hand. Using those with other videos might not work.
2. **linear extrapolation**: linear extrapolation might not always be suitable, especially if we  would also incorporate lane lines further aways
3. **spatial domain only**: I don't incorporate information of the last frames, which might be helpful to further smoothen the results.


###3. Suggest possible improvements to your pipeline

I see lots of potential to improve the pipeline, here are some examples:
1. **Machine Learning for parameter tuning**: if we leave the pipeline like this it would make sense to tune the parameters with training data and furthermore ensure with a validation set that the approach generalizes good enough on new data.
2. **Machine or Deep Learning pipeline**: instead of using traditional computer vision methods it could also make sense to teach a classifier what a lane looks like by just using suitable image training data.
3. **Using other extrapolation methods**: to deal with lane markings further away it could make sense to introduce non-linear regression methods.
3. **Making use of the time domain**: the position of a lane line is highly dependent on the position of the lane in the frame before. Therefore it could make sense to adjust the new detected lane by averaging over past frames.
