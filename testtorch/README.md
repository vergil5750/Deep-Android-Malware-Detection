## Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset


### Requirements

### Data
The <a href="http://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10 dataset</a> consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.  


### Classifier

### Code Description
	1. run_cifar10.lua  - main 
	2. data.lua - load and normalize data
	3. model.lua - define Neural Network architecture; define Loss function
	4. train.lua - train network on training data
	5. test.lua - test network on test data; reporting and plotting

### Experments and metrics
I want to examine:
- GPU .vs. CPU performance
- ReLU .vs. tanh performance, reproduce <a href="img/relu_vs_tanh.jpeg" target="_blank">the graph</a> from <a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf" target="_blank">Krizhevsky et al.</a>
- measure times
- present confusion matrix
- plot error .vs. epoch
- alter weights of trained NN, and without additional training, plot error=f(weight<sub>i,j</sub>)

###CUDA
To use GPU's with torch you call 'require "cutorch"' on a CUDA-capable machine. Here's an explanation of the packages needed for using Torch with GPUs:

    cutorch - Torch CUDA Implementation
    cunn - Torch CUDA Neural Network Implementation
    cunnx - Experimental CUDA NN implementations
    cudnn - NVIDIA CuDNN Bindings


### Results

### Use of AWS 
In case your machine is lack of CUDA-compatible GPU, immidiate and cheap solution could be AWS. Current price is below 10 cent/hour for decent g2.2xlarge instance.
Refer for further instructions: https://github.com/soumith/dl-machine
