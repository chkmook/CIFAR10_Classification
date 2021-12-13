# CIFAR10_Classification

# MACHINE LEARNING 1 (MAT6480.01-00) FINAL PROJECT


## CIFAR-10 Classification

There is a data base called "CIFAR 10" for image classification.
I ask you conduct deep learning program for the classification of CIFAR 10.  Please submit your PYTHON code with the result. You are asked to submit your result to our
TA Dr. Koo, Eunho and his e-mail address is "kooeunho@yonsei.ac.kr".
The due date is December 15.
I like to introduce google search;
The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.


* The CIFAR-10 dataset


The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.


The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.


### Train & Eval
```bash
python train.py
```
can change model (VGG or ResNet) and dataset path at train.py


|configuration||
|-|-|
|Train dataset transform|randomcrop, random horizontal flip|
|Batchsize|32|
|loss|cross entropy loss|
|optimizer|SGD|
|learning_rate|0.01|
|momentum|0.9|
|weight_decay|0.0005|

epoch : 20

|model|Accuracy|
|-----|--------|
|VGG11|83.95|
|VGG13|85.35|
|VGG16|83.59|
|VGG19|83.81|
|ResNet18|77.54|