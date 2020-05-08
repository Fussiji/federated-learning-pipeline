import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import copy
import numpy as np

import syft as sy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_Settings():
    def __init__(self):
        # local training
        self.batch_size = 16
        self.lr = 0.01
        self.local_epochs = 2
        self.loss = F.nll_loss
        # pipeline
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.test_batch_size = 1000
        self.epochs = 3
        self.vworkers = 10
        self.p_available = 0.2    # probability a virtual worker is available
        self.save_model = False

    # create datasets
    def gen_data(self):
        mnist_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))

        mnist_test = datasets.MNIST('../data', train=False, 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        
        return mnist_train, mnist_test

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
            vworker_optimizers.append(optim.SGD(worker_model.parameters(), lr=self.lr))

        return vworker_models, vworker_params, vworker_optimizers

    def worker_availability(self, epoch):
        #vworker_avail = np.random.choice(2, self.vworkers, p=[1-self.p_available, self.p_available])
        vworker_avail = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return vworker_avail


