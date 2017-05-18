**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Instead I applied a Deep Learning approach since the results of the HOG-based approach wasn't satisfying enough
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/mosaic_1.png
[image2]: ./examples/mosaic_2.png
[image3]: ./examples/masked_picture_1.png
[image4]: ./examples/masked_picture_2.png
[image5]: ./examples/mask_1.png
[image6]: ./examples/mask_2.png
[image7]: ./examples/mask_3.png
[image8]: ./examples/detection_1.png
[image9]: ./examples/detection_2.png
[image10]: ./examples/detection_3.png
[video1]: ./final_submission.mp4

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Deep Learning vehicle detection

#### 1. Explain how (and identify where in your code) you created training data for your approach.

The code for this step is contained in code cells 2-4 of the IPython notebook called submission.ipynb.
On the one side I used the CrowdAI training data provided by Udacity. Here I resized the images to ca. 0.25% and used the bounding box annotations to create masks. Every mask contains a 1 if a car is present and a 0, if no car is present.

![alt text][image3]

![alt text][image4]

I also wanted to make use of the KTTI data. As is, it can't be directly used for training a U-Net. We directly have the crops of cars and non-cars, which could be used for example if using Fully Convolutional Networks. In the U-Net setting we need masks. I therefore create mosaics of random non-car images and place one resized random car crop somewhere in the image. Accordingly the masks (y_train) are produced by creating a np.zeros array and only setting value=1 where we put the car. The results don't look realistic but seem to help the model to generalize quite well.

![alt text][image1]

![alt text][image2]

#### 2. Explain how you settled on your final choice of training data parameters.

I first started to work with grayscale images, but recognized during Deep Learning training that the models performs much better, when using colored images. I therefore decided to keep all 3 channels.
I furthermore only worked with the CrowdAI data in the beginning. Since the results of the U-Net weren't good enough, I thought about a potential way to incorporate the KTTI approach. I described this process in step 1.

#### 3. Describe how (and identify where in your code) you trained a classifier.

I trained a U-Net in segment 3 of the Jupyter notebook. As already described I use an approach totally relying on U-Net and used code examples from https://github.com/jocicmarko/ultrasound-nerve-segmentation to build the model in Keras. For this a custom loss function is implemented, which should get quite close to the intersection over union

```
def iou_loss(y_true, y_pred):
    iou_simple = 2 * K.sum(K.flatten(y_true) *K.flatten(y_pred))
    return -(iou_simple / (K.sum(K.flatten(y_true))+K.sum(K.flatten(y_pred))))
```

The model architecture itself is based on https://github.com/jocicmarko/ultrasound-nerve-segmentation and consists of an encoder-decoder structure, where the Upsamling layers of the decoder are concatenated with the Convolution-layers of the encoder. For optimization I used Adam with a learning rate of 1e-4.

### Heatmap creation

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For U-Net no sliding window search is necessary, you simply predict the output mask. During training I looked at the whole image. I then also predicted on the whole image but filtered out every potential detection in the upper half of the image.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried grayscale and colored images. I furthermore experimented with different optimization algorithms, where RMSprop didn't work at all and Adam worked quite decently. Training the U-Net was extremely sensitive regarding the learning rate and took me quite some manual tweaking. I also played around with different approaches to create additional training data and finally ended up with the mosaic approach explained above.

Here are some example images for the mask generation:

![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The heatmap was directly created during prediction.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I used the code provided in the lecture. Since the model was trained on vehicle images, I could assume that each blob corresponds to a vehicle. Therfore bounding boxes were constructed over the blobs.

Here are three heatmaps and their corresponing bounding boxes:

![alt text][image8]
![alt text][image5]

![alt text][image9]
![alt text][image6]

![alt text][image10]
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Eventhough the approach based on HOG worked quite well, I wasn't satisfied. I therefore wanted to finally get my hands dirty with U-Net, which I hadn't had experience with before. And to be honest, I was quite impressed with the results. Even if I just used 2.500 training images the classifier could perform very well on the project video.

What I recognized anyway, that I still have some false positives. I see two reasons for the lack of enough generalization on the new data:
- First, I only used 2.500 training instances. By increasing the amount of training images, I can imagine that the U-Net would perform even better.
- I didn't really use any augmentation at all. By using augmentation, the performance on the unseen video could definetely be improved as well.
