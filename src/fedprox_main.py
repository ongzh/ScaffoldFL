#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import gc
from tensorboardX import SummaryWriter

from args import args_parser
from updates import ProxUpdate, test_results
from models import cifarCNN, VGG
from utils import get_dataset, exp_details, average_weights
import torchvision.models as models

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        # if args.gpu_id:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'cifar':
            global_model = cifarCNN(args=args)

    elif args.model == 'vgg':
        if args.dataset == 'cifar' and args.pretrained:
            global_model = models.vgg16(pretrained=True)
            # change the number of classes
            global_model.classifier[6].out_features = 10
            # freeze convolution weights
            for param in global_model.features.parameters():
                param.requires_grad = False
        elif args.dataset == 'cifar':
            global_model = VGG(args=args)
        else:
            exit(args.dataset + ' with ' + args.model + ' not supported')

    elif args.model == "resnet18":
        global_model = models.resnet18(pretrained=True)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # Test each round
    test_acc_list = []

    #devices that participate
    m = max(int(args.frac * args.num_users), 1)


    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Local Epochs list to account for stragglers
        if args.stragglers == 0:
            local_epoch_list = np.array([args.local_ep] * m)
        else:
            straggler_size = int(args.stragglers * m)
            local_epoch_list = np.random.randint(1, args.local_ep, straggler_size)

            remainders = m - straggler_size
            rem_list = [args.local_ep] * remainders

            epoch_list = np.append(local_epoch_list, rem_list, axis=0)
            # shuffle the list and return
            np.random.shuffle(local_epoch_list)

        global_model.train()

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx, local_epoch in zip(idxs_users, local_epoch_list):
            local_model = ProxUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, local_epoch=local_epoch)
            w, loss, time = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            gc.collect()
            torch.cuda.empty_cache()

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        for c in range(args.num_users):
            local_model = ProxUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, local_epoch=args.local_ep)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            gc.collect()
            torch.cuda.empty_cache()

        train_accuracy.append(sum(list_acc)/len(list_acc))

        round_test_acc, round_test_loss = test_results(
            args, global_model, test_dataset)
        test_acc_list.append(round_test_acc)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Test Accuracy at round ' + str(epoch+1) +
                  ': {:.2f}% \n'.format(100*round_test_acc))

    # Test inference after completion of training
    test_acc, test_loss = test_results(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}]_u[{}]_%strag[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr,args.u, args.stragglers)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    #print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # save results to csv
    res = np.asarray([test_acc_list])
    res_name = '../save/csvResults/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}]_u[{}]_%strag[{}].csv'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr, args.u, args.stragglers)
    np.savetxt(res_name, res, delimiter=",")

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                     args.iid, args.local_ep, args.local_bs))

    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy , color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #              format(args.dataset, args.model, args.epochs, args.frac,
    #                     args.iid, args.local_ep, args.local_bs))

    # Plot Test Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), test_acc_list, color='k')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylim([0.1, 1])
    plt.savefig('../save/fedPROX_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}]_mu[{}]_%strag[{}]_test_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.mu, args.stragglers))

