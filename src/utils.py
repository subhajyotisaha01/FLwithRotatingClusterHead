#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torchvision import transforms
# print("D10")
from torchvision import datasets
# print("D9")
import copy
# print("D2")
import torch
# print("D4")
from torchvision import datasets, transforms
# print("D3")
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


# print("DU1")
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def reinitialize_last_layers( w ) :
    """
    Returns the weights after reinitializing the last 3 layers ( mosly Fully Connected Layers ) to Zero
    """
    # print('HIII')
    # print(tuple(w['fc1.weight'].size()))
    w_new = copy.deepcopy(w)
    w_new['fc1.weight'] = torch.rand( tuple( w['fc1.weight'].size() ) )
    w_new['fc1.bias'] = torch.rand( tuple( w['fc1.bias'].size() ) )
    w_new['fc2.weight'] = torch.rand( tuple(  w['fc2.weight'].size() )  )
    w_new['fc2.bias'] = torch.rand( tuple( w['fc2.bias'].size() ) )
    # for l in range( n_layers, num_of_layers_to_keep ):
    #     bias_shape = client_weights[2 * l].shape 
    #     weight_shape = client_weights[ 2 * l - 1 ].shape
    #     client_weights[2 * l] = np.random.rand( bias_shape )
    #     client_weights[2 * l -1] = np.random.rand( weight_shape )    

    return w


# def cluster_head_data(clustered_users, current_cluster_head_idx, user_groups, num_clusters, num_users_per_cluster):
  
#     cluster_heads_data_idx = {i: set() for i in range(num_clusters)}
#     for i in range(num_clusters):
#         cluster_heads_data_idx[i] = cluster_heads_data_idx[i].union(user_groups[current_cluster_head_idx[i]])
#         for j in range(num_users_per_cluster):
#             if clustered_users[j] != current_cluster_head_idx[i] :
#                 data_size = len(user_groups[clustered_users[j]])
#                 transfer_data_size = int((20/100)*data_size)
#                 cluster_heads_data_idx[i] = cluster_heads_data_idx[i].union(np.random.choice(user_groups[clustered_users[j]], size = transfer_data_size, replace = False))
#     return cluster_heads_data_idx


def exp_details(args):
    print('\nExperimental details:')
    # print(f' Args : {args}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
