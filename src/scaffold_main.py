#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
from torch.utils.tensorboard import SummaryWriter

from args import args_parser
from updates import test_results, ScaffoldUpdate
from models import cifarCNN
from utils import get_dataset, exp_details

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
    
    #only one model for now
    if args.dataset == 'cifar':
        global_model = cifarCNN(args=args)
        control_global = cifarCNN(args=args)

    #set global model to train
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    control_global.to(device)
    
    control_weights = control_global.state_dict()
    
    

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # Test each round
    test_acc_list = []
    
    
    #devices that participate (sample size)
    m = max(int(args.frac * args.num_users), 1)
    
    #model for local control varietes
    local_controls = [cifarCNN(args=args) for i in range(args.num_users)]
    #local_models = [cifarCNN(args=args) for i in range(args.num_users)]
    
    for net in local_controls:
        net.load_state_dict(control_weights)
    
        
    #initiliase total delta to 0 (sum of all control_delta, triangle Ci)
    delta_c = copy.deepcopy(global_model.state_dict())
    #sum of delta_y / sample size
    delta_x = copy.deepcopy(global_model.state_dict())
    
    

    
    
    #global rounds
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_x:
            delta_x[ci] = 0.0
    
        global_model.train()
        # sample the users 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = ScaffoldUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            weights, loss , local_delta_c, local_delta, control_local_w, _ = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, control_local
                = local_controls[idx], control_global = control_global)

            if epoch != 0:
                local_controls[idx].load_state_dict(control_local_w)
            
            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(loss))
            
            #line16
            for w in delta_c:
                if epoch==0:
                    delta_x[w] += weights[w]
                else:
                    delta_x[w] += local_delta[w]
                    delta_c[w] += local_delta_c[w]
            
            #clean
            gc.collect()
            torch.cuda.empty_cache()
        
        #update the delta C (line 16)
        for w in delta_c:
            delta_c[w] /= m
            delta_x[w] /= m
        
        #update global control variate (line17)
        control_global_W = control_global.state_dict()
        global_weights = global_model.state_dict()
        #equation taking Ng, global step size = 1
        for w in control_global_W:
            #control_global_W[w] += delta_c[w]
            if epoch == 0:
                global_weights[w] = delta_x[w]
            else:
                global_weights[w] += delta_x[w]
                control_global_W[w] += (m / args.num_users) * delta_c[w]

        

           
        
            
        
        #update global model
        control_global.load_state_dict(control_global_W)
        global_model.load_state_dict(global_weights)
        
        #########scaffold algo complete##################
        
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        
        global_model.eval()

        for c in range(args.num_users):
            local_model = ScaffoldUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            #print("user:" + str(c) +" " + str(acc))
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
    
    # save results to csv
    res = np.asarray([test_acc_list])
    res_name = '../save/csvResults/Scaffold_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_LR[{}].csv'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.lr)
    np.savetxt(res_name, res, delimiter=",")
