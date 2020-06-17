# Robust-ResNet
IJCAI2019 paper: Towards Robust ResNet: A Small Step but a Giant Leap

<https://arxiv.org/abs/1902.10887>

## Packages needed:
Python3.5, numpy, pytorch, matplotlib, pickle, absl, json, sklearn, and tqdm.  
In order to run our codes, A GPU is required for the speed. 


## Run codes
(1) git clone the directory into local machine.
(2.1) Run code on CIFAR-10
```
chmod +x ./cifar-10_torch/run.sh

./cifar-10_torch/run.sh

python3 ./cifar-10_torch/vis_acc.py # run this code for visualizing its accuracy
```

(2.2) Run code on AG-NEWS
```
chmod +x ./ag-news_torch/run.sh

./ag-news_torch/run.sh  # After running, you can visualize accuracy over epochs in the txt file.
```

## Try different configurations
You can also try various configurations with different hyperparameter settings, e.g. learning rate, epochs, depth (n), different ways of initializations, and optimizers. In addition, you can also remove Batch Normalization layers to have fun. 


## Reference
Please cite our paper if find our code useful. 

```
@inproceedings{zhangj_robust_resnet,
  title     = {Towards Robust ResNet: A Small Step but a Giant Leap},
  author    = {Zhang, Jingfeng and Han, Bo and Wynter, Laura and Low, Bryan Kian Hsiang and Kankanhalli, Mohan},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {4285--4291},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/595},
  url       = {https://doi.org/10.24963/ijcai.2019/595},
}

```

## Contact

Please contact j-zhang@comp.nus.edu.sg, if you have any questions on the codes.
