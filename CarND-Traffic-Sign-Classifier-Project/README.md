# **Traffic Sign Recognition** 

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

For more detail refer to Link: [Medium Article](https://medium.com/@garvtambi05/self-driving-car-traffic-sign-classifier-udacity-p3-677889288127) or join [myWhatsappGroup](https://chat.whatsapp.com/LEO0HxBQd3BBkG54veH00H)


# Article
I wrote an article about the steps involved in this algorithm as well as my experience with applying it on real-world self-collected data.Please refer to  

Link: [Medium Article](https://medium.com/@garvtambi05/self-driving-car-traffic-sign-classifier-udacity-p3-677889288127)

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
3. Follow the instructions in the `Traffic_Signs_Classifier.ipynb` notebook.


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The dataset contains:-
* Number of training examples = 34799
* Number of valid examples= 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### Preprocessing the image data

Although colors in the traffic sign are important in real-world for
people to recognize different signs, traffic signs are also different
in their shapes and contents. We can ignore colors in this problem
because signs in our training set are differentiable from their
contents and shapes, and the network seems to have no problem to learn
just from shapes.
Therefore, My preprocessing phase normalizes images from [0, 255] to
[0, 1], and grayscales it.
Minimally, the image data should be normalized so that the data has a mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data.

See the result in, [notebook](https://github.com/GarvTambi/Udacity-Self-Driving-Car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

![alt text][image1]

### Final Model Architecture

I adapted LeNet architecture: Two convolutional layers followed by one
flatten layer, drop out layer, and three fully connected linear
layers.

* 5x5 convolution (32x32x1 in, 28x28x6 out)
* ReLU
* 2x2 max pool (28x28x6 in, 14x14x6 out)
* 5x5 convolution (14x14x6 in, 10x10x16 out)
* ReLU
* 2x2 max pool (10x10x16 in, 5x5x16 out)
* 5x5 convolution (5x5x6 in, 1x1x400 out)
* ReLu
* Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
* Concatenate flattened layers to a single size-800 layer
* Dropout layer
* Fully connected layer (800 in, 43 out)

#### Type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I train the model in 60 iterations (epochs), and each iteration is
trained with 50 batch size. Adam optimizer is used with learning rate
0.0009.


#### Results on the training, validation and test sets

My final model results were:-
* training set accuracy of 0.97 (overfitting the cross-validation)
* validation set accuracy of 0.954
* test set accuracy of 0.100
The first model is adapted from LeNet architecture. Since LeNet
architecture has a great performance on recognizing handwritings, I
think it would also work on classifying traffic signs.
I used the same parameter given in the LeNet lab. Its training accuracy
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
I have my explanations of the effect of the drop out layer after I've
seen some of the training data. Some images are too dark to see the
sign, so it seems that these images act as noises in the training data
, and the drop out layer can reduce the negative effects on learning.
The final accuracy in the validation set is around 0.954.

#### Parameter Tuning

Don't Forget to tune your Parameter in your model too for better understanding as well to get a model greater than this.

---
