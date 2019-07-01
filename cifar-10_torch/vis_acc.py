#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('ggplot')
#plt.rcParams["font.family"] = "Times New Roman"


# In[ ]:


start_LR = 0.1
num_epochs = 80
epoch = [i for i in range(num_epochs)]

seed_list = [7
            #8
            #9
            #10
            #11
            ] # repeated files you are running
depth_list = [
    #20, 
    #38, 
    #44, 
    #56, 
    110, 
    #134, 
    #152, 
    #182, 
    #242, 
    #302, 
    #326, 
    #434, 
    #542, 
    #650
]

for depth in depth_list:
    h_01_best_acc.update({depth: []})
    h_10_best_acc.update({depth: []})
    
    for seed in seed_list:
        h = 0.1
        name = "seed{}depth{}epoch{}h{}start_LR{}optimzer{}".format(seed, depth, num_epochs, h, start_LR,"SGD_momentum")
        with open( "./vis_acc/" + name + ".pickle", 'rb') as file:
            stat = pickle.load(file)
        test_acc = stat['test_acc']
        plt.plot(epoch, test_acc, color = 'b')
        h_01_best_acc[depth].append(max(test_acc))
        
       
    for seed in seed_list:
        h = 1.0
        name = "seed{}depth{}epoch{}h{}start_LR{}optimzer{}".format(seed, depth, num_epochs, h, start_LR,"SGD_momentum")
        with open( "./vis_acc/" + name + ".pickle", 'rb') as file:
            stat = pickle.load(file)
        test_acc = stat['test_acc']
        plt.plot(epoch, test_acc, color = 'r')

        h_10_best_acc[depth].append(max(test_acc))
    
    plt.show()

#depth_list = [20, 38, 44, 56, 110, 134, 152, 182, 242, 302, 326, 434, 542, 650]

median_h01 = []
std_h01 = []
median_h10 = []
std_h10 =[]
for depth in depth_list:
    median_h01.append(np.median(h_01_best_acc[depth]))
    std_h01.append(np.std(h_01_best_acc[depth]))
    
    median_h10.append(np.median(h_10_best_acc[depth]))
    std_h10.append(np.std(h_10_best_acc[depth]))

    
    

    


# In[ ]:


# Collect some statistics 
median_h01 = np.array(median_h01)
std_h01 = np.array(std_h01)
median_h10 = np.array(median_h10)
std_h10 =np.array(std_h10)

