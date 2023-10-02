import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from GameFormer.predictor import GameFormer
from torch.utils.data import DataLoader
from GameFormer.train_utils import *


'''
This script provides a simple forward pass of the GameFormer model.
'''



def model_training():

    # set seed
    set_seed(args.seed)

    # set up model
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    gameformer = gameformer.to(args.device)
    logging.info("Model Params: {}".format(sum(p.numel() for p in gameformer.parameters())))

    # set up optimizer
    optimizer = optim.AdamW(gameformer.parameters(), lr=args.learning_rate)
    batch_size = args.batch_size

    epoch_loss = []
    epoch_metrics = []
    gameformer.train()
    
    # set up data loaders
    train_set = DrivingData(args.train_set + '/*.npz', args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    batch = next(iter(train_loader))
    # prepare data
    inputs = {
        'ego_agent_past': batch[0].to(args.device), # [batch, 21, 7]
        'neighbor_agents_past': batch[1].to(args.device), # [batch, 20, 21, 7]
        'map_lanes': batch[2].to(args.device), # [batch, 40, 50, 7]
        'map_crosswalks': batch[3].to(args.device), # [batch, 5, 30, 3]
        'route_lanes': batch[4].to(args.device) # [batch, 10, 50, 3]
    }

    ego_future = batch[5].to(args.device)
    neighbors_future = batch[6].to(args.device)
    neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

    # call the mdoel
    optimizer.zero_grad()
    ################### 1. forward pass ###################
    level_k_outputs, ego_plan = gameformer(inputs)
    ########################################################
    loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
    prediction = results[:, 1:]
    plan_loss = planning_loss(ego_plan, ego_future)
    loss += plan_loss

    # loss backward
    loss.backward()
    nn.utils.clip_grad_norm_(gameformer.parameters(), 5)
    optimizer.step()

    # compute metrics
    metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
    epoch_metrics.append(metrics)
    epoch_loss.append(loss.item())
    # data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))


    

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train data', default="/data1/nuplan/jiale/exp/GameFormer/example_data")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--encoder_layers', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_levels', type=int, help='levels of reasoning', default=2)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    # parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    model_training()

'''
python train.py \
--train_set /data1/nuplan/jiale/exp/GameFormer/example_data \
'''