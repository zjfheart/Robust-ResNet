# Robust-ResNet
Towards Robust ResNet: A Small Step but a Giant Leap


## Packages needed:
Python3.5, numpy, pytorch, matplotlib, pickle, absl.

In order to play with our codes, A GPU is required.


## Run codes
(1) git clone the directory into local machine.
(2.1) ## Run code on CIFAR-10
chmod +x ./cifar-10_torch/run.sh

./cifar-10_torch/run.sh

python3 ./cifar-10_torch/vis_acc.py # run this code for visualizing its accuracy


(2.2) ## Run code on AG-NEWS
chmod +x ./ag-news_torch/run.sh

./ag-news_torch/run.sh  # After running, you can visualize accuracy over epochs in the txt file.


## Try different configurations
You can also try various configurations with different hyperparameter settings, e.g. learning rate, epochs, depth (n), different ways of initializations, and optimizers. In addition, you can also remove Batch Normalization layers to have fun.  
