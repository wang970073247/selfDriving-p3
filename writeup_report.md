
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

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
First,i use a Keras lambda layer to normalize the data(model.py line 43 ).
Next,the model consists of 5 convolution neural networks(model.py lines 47-51),they are:
  CNN layer1:filter sizes(5x5), stride(2x2), activation(relu), depth(24) 
  CNN layer2:filter sizes(5x5), stride(2x2), activation(relu), depth(36) 
  CNN layer3:filter sizes(5x5), stride(2x2), activation(relu), depth(48)
  CNN layer4:filter sizes(3x3), activation(relu), depth(64) 
  CNN layer5:filter sizes(3x3), activation(relu), depth(64)
Third,the model has a flatten layer and four fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 62 ). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track,and i try different values of the epoch parameter to avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 61-62), and to get better model i use the right images and left images from multiple cameras,also with a correction parameter which can be turned

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the udacity's data(data.zip),besides i also pick some more training data and add them to the udacity's data. Because when i only use udacity's data, i find the car cannot run well in one of the three sharp corners and on the bridge. so i collect some data when the car in the corner and on the bridge, and i also make the car run counter-clockwise, pick one more round data.

I also used a combination of center lane driving, recovering from the left and right sides of the road.

I also used a multiple cameras data.


