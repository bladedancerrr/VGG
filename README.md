# VGG_Implementation

VGG is a convolutional neural network architecture built for image classification.
It consists of 

1. Convolution + ReLU
2. Max Pooling
3. Fully connected + ReLU
4. Softmax


In this project, I tried to implement a slightly modified version of VGG and trained it on the CIFAR-10 image dataset of 50,000 training and 10,000 test images of 10 different classes. 

Modification: CIFAR-10's images are too small that after the last max-pool, size becomes 1x1. So instead of adding fully 
connected layers, we go straight to a 1x1 convolutional layer.
