## Zhang Jingfeng 
## Email: jingfeng.zhang9660@gmail.com

# import lib
import os
import sys
from absl import flags
import re
import torch
import itertools
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
from sklearn import utils, metrics
import json
from data_helper_noisy import *

 # generate random numbers on all GPs

# Parameters settings
#GPU specify
flags.DEFINE_integer("GPU", "0", "specify GPU to use")

flags.DEFINE_string("database", "ag_news", "specify database" ) 
flags.DEFINE_integer('nthreads', 4, "the number of thread to load data from iterators") 
flags.DEFINE_integer("classes", 4, "Number of categories of this dataset. default ag_news 4") 


# Model Hyperparameters
flags.DEFINE_integer("sequence_length", 1024, "Sequence Max Length (default: 1024)")
flags.DEFINE_string("pool_type", "max", "Types of downsampling methods, use either three of max (maxpool), k_max (k-maxpool) or conv (linear) (default: 'max')")
flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47, 64, 68 (default: 9)")
flags.DEFINE_boolean("shortcut", True, "Use optional shortcut (default: True)")
flags.DEFINE_boolean("BN", True, "Whether to open batch normalization after conv lalyers (default : True)" )
flags.DEFINE_float("h", 0.1, "Identity mapping parameters, step size h. (we can also call this smoother, smaller the model is smoother)") 

# Training parameters
flags.DEFINE_integer("batch_size", 128, "Batch Size")
flags.DEFINE_integer("num_epochs", 15, "Number of training epochs")
flags.DEFINE_float("noise_level", 0.0, "The noise injected into training input feature x")
#flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on test set after this many steps (default: 100)")
flags.DEFINE_float("start_lr", 0.1, "The start learning rate, devided by 10 at every 20 epochs")
flags.DEFINE_integer("lr_halve_interval", 100, "Number of epochs before shrinking learning rate by gamma (0.9)" ) 
flags.DEFINE_float("gamma", 0.9, "the gamma change the lr scheduler") 

# saving model parameters
flags.DEFINE_integer("snapshot_interval", 2, "save model every epoches interval") 


## Seed
flags.DEFINE_integer("seed", 7, "different seed for repeated experiments")

FLAGS = flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
print("-"*20)
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")


seed_num = FLAGS.seed
np.random.seed(seed_num)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

dataset_load = FLAGS.database
database_path = "{}_csv/".format(dataset_load)
n_classes = FLAGS.classes
depth = FLAGS.depth
maxlen = FLAGS.sequence_length
shortcut = FLAGS.shortcut
batch_size = FLAGS.batch_size
epochs = FLAGS.num_epochs
lr = FLAGS.start_lr
lr_halve_interval = FLAGS.lr_halve_interval  # "Number of iterations before halving learning rate"
snapshot_interval = FLAGS.snapshot_interval
gamma = FLAGS.gamma
gpuid = FLAGS.GPU
nthreads = FLAGS.nthreads
BN= FLAGS.BN
h= FLAGS.h
noise_level=FLAGS.noise_level

if BN:
    print ("we are using BN------")
    from net import *
else:
    print ("we are NOT using BN------")
    from net_wobn import *
    
    
## we are adding noise to the original text and then make it to the vector. 
    
data_helper_noisy = data_helper(sequence_max_length=maxlen, noise_level = noise_level, seed = seed_num, mode = "train")
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    train_data_noisy, train_label, test_data, test_label = data_helper_noisy.load_dataset(database_path)
    print("Loading data success...")

    return train_data_noisy, train_label, test_data, test_label

x_train_noisy, y_train, x_test, y_test = preprocess()
    

class Loader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y =y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    
    

def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out/total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] =  num/den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num/den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num/den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] =  num/den if den > 0 else 0

    return dic_metrics




# Train the model

def train(epoch,net,dataset, device,msg="val/test",optimize=False,optimizer=None,scheduler=None,criterion=None):
    
    status = ""
    if optimize:
        net.train()
        status = "train"
    else:
        net.eval()
        status = "test"
    
    #net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx, ty) in enumerate(dataset):
            data = (tx, ty)  # torch.Size([128, 1024])
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()  # zero gradients 

            out = net(data[0])
            ty_prob = F.softmax(out, 1) # probabilites

            #metrics
            y_true = data[1].detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1]

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)  # dic_metrics['accuracy'] = out/total
            
            loss =  criterion(out, data[1])  # This is loss 
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                loss.backward()  # perform a backward pass 
                optimizer.step() # updates the weights
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()
    
    if status == "train":
        results['acc'].append(dic_metrics['accuracy'])
        results['loss'].append(dic_metrics['logloss'] )
    elif status =='test':
        results['val_acc'].append(dic_metrics['accuracy'])
        results['val_loss'].append(dic_metrics['logloss'] )
    

def predict(net,dataset,device,msg="prediction"):
    
    net.eval()

    y_probs, y_trues = [], []

    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):

        data = (tx, ty)  
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


def save(net, txt_dict, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    _ = txt_dict
    torch.save(dict_m,path)


if __name__ == "__main__":
 

    # get data generator
    #tr_loader = DataLoader(Loader(x_train,y_train), batch_size = batch_size, shuffle =True, num_workers=nthreads)
    te_loader = DataLoader(Loader(x_test,y_test), batch_size = batch_size, shuffle =False, num_workers=nthreads)
    tr_noisy_loader = DataLoader(Loader(x_train_noisy,y_train), batch_size = batch_size, shuffle =True, num_workers=nthreads)


    # select cpu or gpu
    device = torch.device("cuda:{}".format(gpuid) if gpuid >= 0 else "cpu")
    list_metrics = ['accuracy']
    print (device)



    # build model
    print("Creating model...")
    # load from database get n_tokens
    n_tokens = 70 
    net = VDCNN(n_classes=n_classes, num_embedding=n_tokens, embedding_dim=16, depth=depth, n_fc_neurons=1024, shortcut=shortcut, h = h)
    criterion = torch.nn.CrossEntropyLoss()


    net.to(device)
    print ("transfered model to devices")


    print(" - optimizer: sgd with momentum")
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9)  


    scheduler = None
    if lr_halve_interval > 0:
        print(" - lr scheduler: {}".format(lr_halve_interval))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_halve_interval, gamma=gamma, last_epoch=-1)



    results = dict()
    results.update({'val_acc': []})
    results.update({'acc': []})
    results.update({'val_loss': []})
    results.update({'loss': []})



    for epoch in range(1, epochs + 1):
        #torch.cuda.empty_cache()
        train(epoch,net, tr_noisy_loader, device, msg="noisy_training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        #torch.cuda.empty_cache()
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)
        #if (epoch % snapshot_interval == 0) and (epoch > 0):
            #path = "{}/model_epoch_{}".format(model_folder,epoch)
            #print("snapshot of model saved as {}".format(path))
            #save(net, variables['txt_dict']['var'], path=path)

    save_name = dataset_load +'_seed_' +str(seed_num)
    save_dir = os.path.join(os.getcwd(), save_name)
    os.makedirs(save_name, exist_ok=True)

    model_name = save_name + "_depth{}h{}BN{}noise_level{}_lr{}".format(depth,h,int(BN),noise_level, lr)
    with open(save_dir+"/"+ model_name + "_results.txt", "w" ) as outfile:
        json.dump(results, outfile)



