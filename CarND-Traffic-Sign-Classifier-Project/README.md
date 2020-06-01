# **Traffic Sign Recognition** 

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

### Submission Files

This project includes

- The notebook `Traffic_Sign_Classifier.ipynb` (and `signames.csv` for completeness)
- `report.html`, the exported HTML version of the python notebook
- A directory `mydata` containing images found on the web
- `README.md`, which you're reading

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

See the result in, [notebook](https://github.com/lijunsong/udacity-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


The code processing images is in cell 9.

Although colors in the traffic sign are important in real world for
people to recoganize different signs, traffic signs are also different
in their shapes and contents. We can ignore colors in this problem
because signs in our training set are differentiable from their
contents and shapes, and the network seems having no problem to learn
just from shapes.

Therefore, My preprocessing phase normalizes images from [0, 255] to
[0, 1], and grayscales it. You can see the grayscale effects in cell
10.


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data

The train, valid and test data are prepreocessed in cell 9. I use
cross validation to split training data. The code to split the data
is in function `train` (see cell 15).

To cross validate my model, I randomly split the given training sets
into training set and validation set. I preserved 10% data for
validation. `sklearn` has the handy tool `train_test_split` to do the
work.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code is in function `classifier` (see cell 11).

I adapted LeNet architecture: Two convolutional layers followed by one
flatten layer, drop out layer, and three fully connected linear
layers.

1. 5x5 convolution (32x32x1 in, 28x28x6 out)
2. ReLU
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. x5 convolution (14x14x6 in, 10x10x16 out)
5. ReLU
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. 5x5 convolution (5x5x6 in, 1x1x400 out)
8. ReLu
9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
10. Concatenate flattened layers to a single size-800 layer
11. Dropout layer
12. Fully connected layer (800 in, 43 out)

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is in cell 15 and 16.

I train the model in 60 iterations (epochs), and each iteration is
trained with 50 batch size. Adam optimizer is used with learning rate
0.0009.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.97 (overfitting the cross validation)
* validation set accuracy of 0.954
* test set accuracy of 0.100


The first model is adapted from LeNet architecture. Since LeNet
architecture has a great performance on recognizing handwritings, I
think it would also work on classifying traffic signs.

I used the same parameter given in LeNet lab. Its training accuracy
initially was around 90%, so I thought the filter depth was not large
enough to capture images' shapes and contents. Previously the filter
depth was 6 for the first layer and 12 for the second. I increased
to 12 and 25. The accuracy increased to around 93%.

I then added a drop out layer, which is supposed to used to prevent
overfitting, but I found a drop out layer could sometimes increase the
accuracy to 95%.

I also tuned `epoch`, `batch_size`, and `rate` parameters, and settled at

- `epoch` 60
- `batch_size` 50
- `learning rate` 0.0009

I have my explainations of the effect of the drop out layer after I've
seen some of the training data. Some images are too dark to see the
sign, so it seems that these images act as noises in the training data
and drop out layer can reduce the negative effects on learning.

The final accuracy in validation set is around 0.954.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I want to see how the classifier performs on similar signs. The
General Caution and Traffic signals: they both look like a vertical bar
(see the visualization) when grayscaled. And pedestrains and child
crossing look similar in low resolution.


Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy on the new traffic signs is 63.6%, while it was 93% on
the test set. This is a sign of underfitting. By looking at the
virtualized result, I think this can be addressed by using more image
preprocessing techniques on the training set.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In the submitted version, the model can correctly guess 7 out of 11
signs. The accuracy is 63.6%. However, it can sometimes predict
correctly 10 out of 11 images.

By looking at the virtualized data. The predictions of pedestrains,
children crossing, and speed limit 60km/h are actually close
enough. This is actually consistent to my various
experiments. Sometimes the prediction accuracy can be as good as
90%. I think to get the consistent correctness, I need more good
data. One simple thing to do might be to preprocess the image by
brightening dark ones. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


