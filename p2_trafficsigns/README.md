# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/examples/1_dist.png "data distribution"
[image2]: ./examples/examples/2_examples.png "image examples"
[image3]: ./examples/examples/2_preproc.png "preprocessing"
[image4]: ./examples/examples/3_dist.png "upsampled distribution"
[image5]: ./examples/examples/5_new.png "new traffic signs"
[image6]: ./examples/examples/4_new.png "preprocessed new traffic signs"
[image7]: ./examples/examples/6_features.png "feature maps"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/datadominik/selfdrivingcar/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate the summary statistics of the dataset.

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth and sixth code cells of the IPython notebook.  

To get an overview about the data distribution I first plotted a bar chart, indicated how many examples we have per traffic sign type. As you can see, we have a quite unbalanced data distribution.

![alt text][image1]

To see what the images acutally look like I also plotted one example per sign type in a grid. This helped me to understand how the images looks like.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 7th, 8th and 9th code cells of the IPython notebook.

As a first step I coverted all images to grayscale by taking the mean of all three color channels. This showed to work quite well but you need to keep it mind, that for German traffic signs color is actually a strong feature.

Next I applied zero-mean and unit-variance normalization since it has shown to lead to faster convergence when training Deep Neural Networks. This step is not necessary but helped in this experimental setup

You can see the pre-processing process in the following image:
![alt text][image3]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data used was already split into training, testing and validation sets. Therefore no further effort was put into this topic. 

But the dataset is highly imbalanced, therefore I implemented two methods used for upsampling: 
* rotational augmentation: rotate the images to achieve some sort of rotational invariance later
* scaling augmentation: zoom-in/ zoom-out, to achieve some sort of scale invariance later

The augmentation was performed to have each class at least represented by 1000 examples. Therefore the class distribution of the augmented dataset looks like this: 

![alt text][image4]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 13th cell of the ipython notebook. I was inspired by the LeNet-5 architecture shown in the lecture but tinkered around to create something slightly different. The model consists of two conv-relu-maxpooling blocks, followed by two hidden fully connected layers, before we end in the last softmax layer. The two fully connected layers have dropout regularization to encourage generalization.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image  						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Flatten			    | 1x1 stride, same padding, outputs 800     	|
| Fully connected		| outputs 256   								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 128   								|
| RELU					|												|
| Dropout				|												|
| Softmax				| 43 classes     								|

 

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 14th, 15th and 16th cell of the ipython notebook. 

To train the model, I used the process shown in the lecture but used the RMSprop optimizer instead. I furthermore trained my model on the augmented dataset and used dropout of 0.5 for training. I trained for 10 epochs and used a batch size of 128.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.976
* validation set accuracy of 0.970 
* test set accuracy of 0.946

I started with a simplified version of the architecture I'm using now. I used a 4-layer neural network consisting of 2 convolutional layers and 2 fully-connected layers. Since we're working on images, CNNs are often the best option. I played around with filter sizes between (3,3) and (4,4) but that didn't have much of an effect. Instead I realited that by using 128 filters per layer and not using dropout I ran into overfitting quite fast - the training accuracy exceeded the validation accuracy by a big margin. Therefore I reduced the number of filters and introduced dropout between the fully connected layers. This helped to improve generalization. Varying the batch size also had an effect on the outcome and 128 seemed to work good for this specific problem. After 25 epochs of training I couldn't see much improvement in the validation accuracy and therefore stopped training there. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 

The first image might be difficult to classify. For us it's obvious that we have a stop sign here. But we didn't use shear augmentation to upsample the data and therefore only have front perspectives to train on. The other images should work well, since the these types of image occur a lot in the dataset. By using images where we don't have the nice crop of the image, the network would perform not well from my point of view. It's trained on centered images and and doesn't make use of weight sharing to classify signs in different spots. If we would introduce this invariance in the dataset by further augmentation, the network might be able to deal with it.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th and 20th cell of the Ipython notebook.

After the preprocessing was applied to the pictures, they looked like this:
![alt text][image6] 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Turn left ahead   							| 
| Road work     		| Road work 									|
| 50 km/h				| 50 km/h										|
| No entry	      		| No entry						 				|
| Priority road			| Priority road      							|


4 of 5 traffic signs were predicted correctly. It's hard to compare this accuracy to the test set, since we only use 5 pictures here. But it indicates that the model works well on image data that wasn't part of the initial dataset collection. Having 80% accuracy on this completely new "dataset" indicates some sort of generalization and that we at least didn't do everything wrong. For further justification a much bigger 2nd test dataset would be necessary.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22th and 23th cell of the Ipython notebook.

**True class: Stop**
- Turn left ahead: 0.9329
- Turn right ahead: 0.0454
- No vehicles: 0.0215
- Yield: 0.0001
- Stop: 0.0001

*Unfortunately "Stop" doesn't get detected well. As you can see the classifier seems to be very confident about the turn left ahead sign.*

**True class: Road work**
- Road work: 1.0000
- Dangerous curve to the right: 0.0000
- Bumpy road: 0.0000
- Keep right: 0.0000
- Slippery road: 0.0000

*Road work is the most likely sign. Though the classifier seems to focus on the triangular shape of the sign and therefore picks other signs that share this property as well (even with very low probability).*

**True class: Speed limit (50km/h)**
- Speed limit (50km/h): 0.9697
- Stop: 0.0287
- Speed limit (30km/h): 0.0010
- Speed limit (80km/h): 0.0005
- Speed limit (60km/h): 0.0002

*The model is very confident about the speed limit of 50 km/h and (besides the stop sign) picks other speed limits as its next guesses. Seems reasonable.*

**True class: No entry**
- No entry: 1.0000
- End of all speed and passing limits: 0.0000
- No vehicles: 0.0000
- Stop: 0.0000
- End of no passing: 0.0000

*The model is very confident about its first and correct guess. The next predictions make sense as well and I visualized the filters (at the end of the page). The model picks signs that have strong horizontal bar in the image. Of course the stop sign has letters on it, but they somehow form a centered horizontal bar as well. Looking at the filter visualization you can see the activations in this area of the no entry sign.*

**True class: Priority road**
- Priority road: 1.0000
- Roundabout mandatory: 0.0000
- End of no passing: 0.0000
- No passing: 0.0000
- Right-of-way at the next intersection: 0.0000

*Priority road is with 100% confidence the most probable (and correct) prediction.*

Look at the filter visualizations of the no entry sign: 
![alt text][image7] 