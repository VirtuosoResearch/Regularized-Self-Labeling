from __future__ import print_function, division

# pytorch imports
import torch
from torch._C import device
from torch.autograd.grad_mode import F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E

from constraint import FrobeniusConstraint, LInfLipschitzConstraint, add_penalty

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

def deep_copy(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict.update({k:v.clone().detach()})
    return new_state_dict

def checkpoint(model, best_loss, epoch, LR, name="checkpoint"):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, f'results_{name}/{name}')


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        device,
        name="checkpoint"):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device)).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))
            time_elapsed = time.time() - since
            print('Last for {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, name)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open(f"results_{name}/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(f'results_{name}/{name}')
    model = checkpoint_best['model']

    return model, best_epoch

def add_penalties(norm, lambda_extractor, lambda_pred_head, state_dict=None, scale_factor=1.0):
    penalty = []
    penalty.append(
        {"norm": norm, 
            "_lambda": lambda_extractor,
            "excluding_key": "pred_head",
            "including_key": "layer1",
            "state_dict": state_dict}
    )
    penalty.append(
        {"norm": norm, 
            "_lambda": lambda_extractor,
            "excluding_key": "pred_head",
            "including_key": "layer2",
            "state_dict": state_dict}
    )
    penalty.append(
        {"norm": norm, 
            "_lambda": lambda_extractor*scale_factor,
            "excluding_key": "pred_head",
            "including_key": "layer3",
            "state_dict": state_dict}
    )
    penalty.append(
        {"norm": norm, 
            "_lambda": lambda_extractor*pow(scale_factor, 2),
            "excluding_key": "pred_head",
            "including_key": "layer4",
            "state_dict": state_dict}
    )
    penalty.append(
        {"norm": norm, 
            "_lambda": lambda_pred_head,
            "excluding_key": None,
            "including_key": "pred_head",
            "state_dict": None}
    )
    return penalty

def add_constraint(model_type, norm, lambda_extractor, lambda_pred_head, state_dict = None, scale_factor = 1.0, use_ratio = False):
    '''
    Add hard constraint for model weights
        for feature_extractor, it will contraint the weight to pretrain weight
        for pred_head, it will contraint the weight to zero
    '''
    constraints = []
    if norm == "inf-op":
        constraints.append(
            LInfLipschitzConstraint(model_type, lambda_extractor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
        )
        constraints.append(
            LInfLipschitzConstraint(model_type, lambda_extractor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
        )
        constraints.append(
            LInfLipschitzConstraint(model_type, lambda_extractor*scale_factor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
        )
        constraints.append(
            LInfLipschitzConstraint(model_type, lambda_extractor*pow(scale_factor, 2), 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
        )
        constraints.append(
            LInfLipschitzConstraint(model_type, lambda_pred_head, 
            including_key = "pred_head")
        )
    elif norm == "frob":
        constraints.append(
            FrobeniusConstraint(model_type, lambda_extractor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
        )
        constraints.append(
            FrobeniusConstraint(model_type, lambda_extractor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
        )
        constraints.append(
            FrobeniusConstraint(model_type, lambda_extractor*scale_factor, 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
        )
        constraints.append(
            FrobeniusConstraint(model_type, lambda_extractor*pow(scale_factor, 2), 
            state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
        )
        constraints.append(
            FrobeniusConstraint(model_type, lambda_pred_head, including_key = "pred_head")
        )
    return constraints

def train_model_constraint(
    model,
    criterion,
    optimizer,
    LR,
    num_epochs,
    dataloaders,
    dataset_sizes,
    weight_decay,
    device,
    constraints = [], penalty = [], name = "checkpoint", label_smooth = False, smooth_alpha = 0.2):
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device)).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if label_smooth:
                    smooth_target = 0.5*torch.ones_like(outputs).detach()
                    loss += smooth_alpha * torch.nn.functional.binary_cross_entropy(outputs, smooth_target)
                            # torch.mean(0.5*torch.log2(outputs) + 0.5*torch.log2(1-outputs))
                """Apply Penalties"""
                for penal in penalty:
                    loss += add_penalty(
                        model, 
                        penal["norm"], 
                        penal["_lambda"], 
                        excluding_key = penal["excluding_key"],
                        including_key = penal["including_key"],
                        state_dict=penal["state_dict"]
                    )
                """Apply Penalties"""

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    """Apply Constraints"""
                    for constraint in constraints:
                        model.apply(constraint)
                    """Apply Constraints""" 

                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))
            time_elapsed = time.time() - since
            print('Last for {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, name)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open(f"results_{name}/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(f'results_{name}/{name}')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY, args, lambda_extractor = 1.0, lambda_pred_head = 1.0, scale_factor = 1.0, name = "checkpoint"):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 16

    try:
        rmtree(f'results_{name}/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs(f"results_{name}/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    # model = models.densenet121(pretrained=True)
    # num_ftrs = model.classifier.in_features
    # # add final layer with # outputs in same dimension of labels with sigmoid
    # # activation
    # model.classifier = nn.Sequential(
    #     nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
    device = f"cuda:{args.device}"
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    constraints = []
    if args.reg_method == "constraint":
        constraints = add_constraint(
            model_type = type(model), norm = args.reg_norm, 
            lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
            state_dict = source_state_dict, scale_factor=scale_factor
        )
    penalty = []
    if args.reg_method == "penalty":
        penalty = add_penalties(
            norm = args.reg_norm, lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
            state_dict = source_state_dict, scale_factor=scale_factor
        )

    # train model
    model, best_epoch = train_model_constraint(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY, device = device,
                                    constraints = constraints, penalty = penalty, name = name,
                                    label_smooth=args.label_smooth, smooth_alpha=args.smooth_alpha)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES, device=device)

    return preds, aucs
