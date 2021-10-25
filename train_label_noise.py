import os
import argparse
import collections
from numpy.core.fromnumeric import argsort
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import ConstraintTrainer, LossReweightConstraintTrainer
from utils import prepare_device, deep_copy
from data_loader.random_noise import label_noise, noisy_labeler
import time

def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

    device, device_ids = prepare_device(config['n_gpu'])
    if args.noisy_labeler:
        aux_model = module_arch.ResNet18(
            n_classes=config["arch"]["args"]["n_classes"]-1 if args.train_dac else config["arch"]["args"]["n_classes"]
        )
        aux_model.load_state_dict(torch.load(args.noisy_labeler_dir)["state_dict"])
        wrong_indices, train_labels_old = noisy_labeler(
            train_data_loader,
            train_data_loader.dataset, 
            train_data_loader.sampler.indices,
            aux_model,
            device = device
        )
        del aux_model
        logger.info("Randomizing {} number of labels".format(wrong_indices.shape[0]))
    elif args.noise_rate!=0:
        wrong_indices, train_labels_old = label_noise(
            train_data_loader.dataset, train_data_loader.sampler.indices, args.noise_rate, symmetric=not args.noise_nonuniform
        )
        logger.info("Randomizing {} number of labels".format(wrong_indices.shape[0]))

    # If small data, shrink training data size
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    accuracies = []
    torch.manual_seed(int(time.time()))
    for run in range(args.runs):
        model.reset_parameters(source_state_dict)
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if args.train_correct_label:
            checkpoint_dir = os.path.join(
            "./saved_label_noise", 
            "{}_{}_noise_rate_{}_correct_label".format(config["arch"]["type"], config["data_loader"]["type"], args.noise_rate))
            trainer = LossReweightConstraintTrainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir,
                        adaptive_epoch=args.reweight_epoch,
                        temp=args.reweight_temp,
                        correct_epoch=args.correct_epoch,
                        correct_thres=args.correct_thres,
                        train_labels_old=train_labels_old)
            lambda_extractor = config["reg_extractor"]
            lambda_pred_head = config["reg_predictor"]
            scale_factor = config["scale_factor"]
            if config["reg_method"] == "constraint":
                trainer.add_constraint(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )
            if config["reg_method"] == "penalty":
                trainer.add_penalty(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )
        else:
            checkpoint_dir = os.path.join(
            "./saved_label_noise", 
            "{}_{}_{}_{}_{:.4f}_{:.4f}_noise_rate_{}".format(config["arch"]["type"], config["data_loader"]["type"], 
                                    config["reg_method"], config["reg_norm"],
                                    config["reg_extractor"], config["reg_predictor"], args.noise_rate))
            trainer = ConstraintTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir = checkpoint_dir)
                        
            lambda_extractor = config["reg_extractor"]
            lambda_pred_head = config["reg_predictor"]
            scale_factor = config["scale_factor"]
            if config["reg_method"] == "constraint":
                trainer.add_constraint(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )
            if config["reg_method"] == "penalty":
                trainer.add_penalty(
                    norm = config["reg_norm"], lambda_extractor = lambda_extractor, lambda_pred_head=lambda_pred_head, 
                    state_dict = source_state_dict, scale_factor=scale_factor
                )

        trainer.train()
        accuracies.append(trainer.test())
    logger.info("Test Accuracy {:1.4f} +/- {:1.4f}".format(np.mean(accuracies), np.std(accuracies)))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--noise_rate', type=float, default=0)
    args.add_argument('--noise_nonuniform', action="store_true")
    args.add_argument('--noisy_labeler', action="store_true")
    args.add_argument('--noisy_labeler_dir', type=str, \
        default="./saved_finetune_models/ResNet18_IndoorDataLoader_none_none_1.0000_1.0000/model_epoch_8.pth")

    args.add_argument('--train_correct_label', action="store_true")
    args.add_argument('--reweight_epoch', type=int, default=5)
    args.add_argument('--reweight_temp', type=float, default=1)
    args.add_argument('--correct_epoch', type=int, default=10)
    args.add_argument('--correct_thres', type=float, default=0.9)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--weight_decay'], type=float, target="optimizer;args;weight_decay"),
        CustomArgs(['--reg_method'], type=str, target='reg_method'),
        CustomArgs(['--reg_norm'], type=str, target='reg_norm'),
        CustomArgs(['--reg_extractor'], type=float, target='reg_extractor'),
        CustomArgs(['--reg_predictor'], type=float, target='reg_predictor'),
        CustomArgs(['--scale_factor'], type=int, target="scale_factor")
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
