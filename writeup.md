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

[image0]: ./visualize.png "Visualization"
[image1]: ./visualize_bar.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_images/01.png "Traffic Sign 1"
[image5]: ./new_images/02.png "Traffic Sign 2"
[image6]: ./new_images/03.png "Traffic Sign 3"
[image7]: ./new_images/04.png "Traffic Sign 4"
[image8]: ./new_images/05.png "Traffic Sign 5"
[trainAccuracy]: ./trainAccuracy.png "Train Accuracy"
[testAccuracy]: ./testAccuracy.png "Test Accuracy"
[validationAccuracy]: ./validationAccuracy.png "Validation Accuracy"
[softmax]: ./softmax.png "Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vivekchand/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic python & numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code for this step is contained in the third code cells of the IPython notebook.
Here is an exploratory visualization of the data set. It pulls in random 10 images and labels them with the corresponding value.

![alt text][image0]


The code for this step is in the fourth cell. At this point I detail the dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed. This helps in creating a balance if required.
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it helps reduce the training time & makes classification easier. The code for this step is in the fifth cell.
The fifth cell also normalizes the grayscale image.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 16x16x64 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 				    |
| Flatten   	      	| input 5x5x16, output 400				        |
| Fully connected		| input 400, output 120      					|
| RELU					|												|
| Droupout				| 50% keep										|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Droupout				| 50% keep										|
| Fully connected		| input 84, output 43							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cell 11 of the ipython notebook.
To train the model, I slightly modified the LeNet model. Added two Dropout layers to refine & runs on 60 EPOCHS with a BATCH_SIZE of 100. 
I used AdamOptimizer at a learning rate of 0.0009. This gave me an accuracy of 96.1%.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.1%
* test set accuracy of 94.6%

![alt text][trainAccuracy]
![alt text][validationAccuracy]
![alt text][testAccuracy]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture I tried was with the original LeNet architecture used for hand written recognition, since the 
dataset is kind of similar.
* What were some problems with the initial architecture?
The accuracy ws not improving until I added a couple of dropout layers & tuned the learning rate, epoch & batch size.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
First, I grayscaled & normalized the images, then added dropout layers just after the fully connected layers. 
Also tuned EPOCH & BATCH_SIZE to 60 & 100 to get a descent accuracy
* Which parameters were tuned? How were they adjusted and why?
EPOCH -- more the iterations accuracy seemed to improve
BATCH_SIZE -- 100 at a times seemed to be right batch size
LEARNING_RATE - 0.0009 although slow seemed to learn best 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Having dropout layers + Going with Normalized Grayscale images helped a lot due to easier classification & quicker to train as well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I used not so clear image, which would be similar to reality when captured on a camera in a moving car.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ![alt text][image4]   | Speed limit (60km/h)   						| 
| ![alt text][image5]   | Speed limit (30km/h)   						| 
| ![alt text][image4]   | Road Work   									| 
| ![alt text][image7]   | Keep Right   									| 
| ![alt text][image8]   | Turn left ahead   							| 


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][softmax]

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .1         			| Speed limit (60km/h)   						| 
| .1     				| Speed limit (30km/h) 							|
| .1					| Road Work										|
| .1	      			| Keep Right					 				|
| .1				    | Turn left ahead      							|



