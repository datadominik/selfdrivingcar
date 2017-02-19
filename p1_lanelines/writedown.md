#**Finding Lane Lines on the Road**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

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
[image2][./examples/transformed.jpg]

Here is video 1 with my lane detection pipeline:
![video1][white.mp4]

And here is video 2 with my lane detection pipeline:
![video2][yellow.mp4]

###2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
