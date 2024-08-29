#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import typing
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import reinitialize_last_layers
import numpy as np

# @typing.runtime_checkable
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        # self.idxs = [int(i) for i in idxs]
        #print(idxs)
        self.idxs = [int(i) for i in idxs]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, users_in_cluster, user_groups, logger):
        self.args = args
        self.logger = logger
        # print(users_in_cluster[0])
        self.users_in_cluster = users_in_cluster
        self.user_groups = user_groups
        self.num_clusters = args.num_clusters
        self.num_users = args.num_users
        self.dataset = dataset
        # num_users_per_cluster = int(args.num_users / args.num_clusters)
        self.current_cluster_head_data_idx = self.cluster_head_data( self.users_in_cluster[0], 
                                                                     user_groups,  args.num_clusters,
                                                                     int(args.num_users / args.num_clusters) )
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(self.current_cluster_head_data_idx))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader( DatasetSplit( dataset, idxs_train ),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(1,self.args.local_ep+1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # 50% of the training is happening at one place and then the cluster head is changed where, only the initial weights are same
            if iter % int(self.args.local_ep / 2) == 0 :
                w = model.state_dict()
                w = reinitialize_last_layers( w )
                model.load_state_dict( w )
                self.current_cluster_head_data_idx = self.cluster_head_data( current_cluster_head_idx = self.users_in_cluster[1], 
                                                                user_groups = self.user_groups, num_clusters = self.num_clusters,
                                                                num_users_per_cluster = int(self.num_users / self.num_clusters) )
                self.trainloader, self.validloader, self.testloader = self.train_val_test(self.dataset, list(self.current_cluster_head_data_idx))



        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def cluster_head_data(self, current_cluster_head_idx , user_groups , num_clusters, num_users_per_cluster):
  
        cluster_heads_data_idx = set()
        # for i in range(num_clusters):
        cluster_heads_data_idx = cluster_heads_data_idx.union(user_groups[current_cluster_head_idx])
        for j in range(num_users_per_cluster):
            if self.users_in_cluster[j] != current_cluster_head_idx :
                data_size = len(user_groups[self.users_in_cluster[j]])
                transfer_data_size = int((5/100)*data_size)
                cluster_heads_data_idx = cluster_heads_data_idx.union(np.random.choice(list(user_groups[self.users_in_cluster[j]]), size = transfer_data_size, replace = False))
        return cluster_heads_data_idx


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
