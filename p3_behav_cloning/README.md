**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/steering.png "Steering angle distribution"
[image2]: ./examples/video.gif "Final drive"
[image3]: ./examples/shadow.jpg "Shadow augmentation"
[image4]: ./examples/dirt.jpg "Dirt augmentation"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists is strongly inspired by the architecture of Nvidia shown in the lecture. It's based on several Convolution-ReLu-MaxPooling blocks followed by a couple of Dense layers, which are separated by Dropout layers. The final layer consists of a single neuron predicting the steering angle. The input_shape of the first layer is (80,160,3) which is 50% of the original image size.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. You can find them in all but the last Dense layer of the model.

Furthermore the available data was split into a training and validation part (90/10). Beforehand random shuffling was applied. After the model training reached a low validation error the simulator was used as final test.

#### 3. Model parameter tuning

The model used adam, so no manual tuning of optimizer parameters was done.

#### 4. Appropriate training data

The training data only consisted of the available data given by Udacity. I tried to train the model on my own driving data, but in fact I wasn't able to drive smooth enough to be a good role model for a self-driving car algorithm.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The strategy for deriving a model architecture consisted of trial and error in the first place.

My first step was to use a pretrained VGG16, where I replaced the Dense layers by new ones. Unfortunately the results weren't as good as expected. I think the fact that the filters of VGG were trained on real images made them too complex for the task here. Both training and validation loss (in this case MSE) were too big and the model failed in the simulator. The simulation images don't have a huge level of detail and therefore a model with much simpler filters could work better.

Therefore I took a step back and tried to use an architecture that is inspired by the original Nvidia model for behavioural cloning. After training this one for several epochs the training and validation loss seemed to decrease constantly.

Since I already decided in the beginning to use augmentation, overfitting was not much of an issue.     

I created the following methods and applied them with a certain probability to the training images:
* Image flip: 0.5
* Add artificial lense dirt: 0.3
* Add shadows: 0.4
* Change color: 0.5

After 4 epochs the car was able to drive some meters without leaving the track. But at some point it always left the road. Looking at the steering angle distribution I realized that we have too much 0 angles and remembered the hint of the teachers: using the left and right images and adding +/- 0.25 to their steering angle to deal with the unbalanced data distribution.

**Data distribution**

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**Final drive**

![alt text][image2]


#### 2. Final Model Architecture
My final model architecture looks like the following. I left out Dropout in the final two Dense layers, maybe adding it there could also further improve the performance.

- Convolution2D: 24, 3, 3
- ReLu
- MaxPooling2D: 2,2
---------------------------
- Convolution2D: 36, 3, 3
- ReLu
- MaxPooling2D: 2,2
---------------------------
- Convolution2D: 48, 3, 3
- ReLu
- MaxPooling2D: 2,2
---------------------------
- Convolution2D: 64, 3, 3
- ReLu
- MaxPooling2D: 2,2
---------------------------
- Dense: 1000
- ReLu
- Dropout: 0.5
---------------------------
- Dense: 500
- ReLu
- Dropout: 0.5
---------------------------
- Dense: 50
- ReLu
---------------------------
- Dense: 10
- ReLu
---------------------------
- Output

#### 3. Creation of the Training Set & Training Process
Even if I tried creating my own data, I relied on the data given by Udacity in the end.

For augmentation I used the methods mentioned above. Attached are some images, where you can see the augmentation effect of the shadowing and noising. Color changes and horizontal flips were also applied to improve performance. Here are two example images (unfortunately very small)

**Shadowing**

![alt text][image3]

```python

def aug_shad(img,y):
    ### adds partial shadows on the camera picture
    ### I've seen similar approaches in several Udacity posts on medium.com
    img = np.copy(img)
    y = np.copy(y)
    ### transform image to hsv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img.astype("float16")
    triangle = np.ones((500,500))
    ### get triangle indices
    xx = np.triu_indices(triangle.shape[0])[0]
    yy = np.triu_indices(triangle.shape[1])[1]
    x_flip = random.choice([-1, 1])
    y_flip = random.choice([-1, 1])
    triangle[x_flip*xx,y_flip*yy] = 0.5
    ### rotate triangle to get different angles
    ### make size afterwards same as the image
    rotation = cv2.getRotationMatrix2D((250+np.random.randint(50),250),np.random.randint(90),1.0)
    triangle = cv2.warpAffine(triangle, rotation, triangle.shape,flags=cv2.INTER_LINEAR)
    triangle = triangle[img.shape[0]-int(img.shape[0]/2):img.shape[0]+int(img.shape[0]/2),img.shape[1]-int(img.shape[1]/2):img.shape[1]+int(img.shape[1]/2)]
    ### mulitply last hsv channel times the triangle to create shadow effect
    img[:,:,2] *= triangle
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return(img,y)
```

**Noising**

![alt text][image4]

```python
def aug_dirt(img,y):
    ### removes small fractions of the image and adds random noise
    ### similar to water drops on camera
    img = np.copy(img)
    y = np.copy(y)
    shape = img.shape
    noise_size = 25
    noise_count = np.random.randint(10)
    for count in range(noise_count):
        row = np.random.randint(shape[0]-noise_size)
        col = np.random.randint(shape[1]-noise_size)
        img[row:row+noise_size,col:col+noise_size] = np.random.randint(255,size=(noise_size,noise_size,3))
    return(img,y)
```

I  randomly shuffled the data set and put 10% of the data into a validation set. The collected training data was normalized by dividing by 255. and then augmented live in the training generator.

I used an adam optimizer to avoid hand-tuning the learning rate. I stopped training after 30 epochs and saved the weights of the model everytime the validation performance increased. The weights used for the final model are therefore the ones with the lowest mean squared error on the validation set.
