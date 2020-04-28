from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

import copy
import numpy as np

import syft as sy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(20, 64)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out')
        self.linear2 = nn.Linear(64, 1)      

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class Stroke_Settings():
    def __init__(self):
        # local training
        self.batch_size = 16
        #self.lr = 0.01
        self.local_epochs = 1
        self.loss = F.binary_cross_entropy #F.nll_loss
        # pipeline
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 100
        self.test_batch_size = 1000
        self.epochs = 10
        self.vworkers = 2
        self.p_available = 1    # probability a virtual worker is available
        self.save_model = False

    # create datasets
    def gen_data(self):
        data = read_csv('./stroke_data/train_2v.csv')
        data['smoking_status'].fillna(method='bfill', inplace=True)
        data['bmi'].fillna(method='bfill', inplace=True)
        dataset = data.values
        X = dataset[:, 1:-1]
        y = dataset[:,-1]
        y = y.reshape((len(y), 1))

        x=[]
        for i in range(X.shape[1]):
            if type(X[0,i])==str:
                x.append(i)
                ohe = OneHotEncoder()
                ohe.fit(X[:,i].reshape(-1, 1))
                t = ohe.transform(X[:,i].reshape(-1, 1)).toarray()
                X = np.append(X,t,axis=1)
        X = np.delete(X,x,axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        print(X_train.shape)
        
        X_train = np.array(list(X_train), dtype=np.float)
        y_train = np.array(list(y_train), dtype=np.float)
        X_test = np.array(list(X_test), dtype=np.float)
        y_test = np.array(list(y_test), dtype=np.float)
        #train = Dataset(X_train, y_train, train=True)
        train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        return train, test

    # federate dataset
    def datafed_method(self, remote_dataset, train_loader, vworker_comp):
        for batch_idx, (data,target) in enumerate(train_loader):
            data = data.send(vworker_comp[batch_idx % self.vworkers])
            target = target.send(vworker_comp[batch_idx % self.vworkers])
            remote_dataset[batch_idx % self.vworkers].append((data, target))

        return remote_dataset

    # aggregate local models
    def aggregate(self, vworker_models, device, vworker_params, vworker_avail):
        new_params = list() # model average

        for param_i in range(len(vworker_params[0])):
            spdz_params = list()

            for remote_index in range(self.vworkers):
                if vworker_avail[remote_index] >= 0.5: # available
                    copy_of_parameter = vworker_params[remote_index][param_i].copy()
                    spdz_params.append(copy_of_parameter)

            new_param = sum(spdz_params) / len(spdz_params) # average
            new_params.append(new_param)

        # update all remote models
        with torch.no_grad():
            for model in vworker_params:
                for param in model:
                    param *= 0

            for remote_index in range(self.vworkers):
                for param_index in range(len(vworker_params[remote_index])):
                    vworker_params[remote_index][param_index].set_(new_params[param_index])

        return vworker_models, vworker_params

    def gen_local(self, device):
        vworker_models = []
        vworker_params = []
        vworker_optimizers = []

        og_model = Net()

        for worker in range(self.vworkers):
            worker_model = copy.deepcopy(og_model).to(device)
            vworker_models.append(worker_model)
            vworker_params.append(list(worker_model.parameters()))
            #vworker_optimizers.append(optim.SGD(worker_model.parameters(), lr=self.lr))
            vworker_optimizers.append(optim.Adam(worker_model.parameters()))

        return vworker_models, vworker_params, vworker_optimizers


