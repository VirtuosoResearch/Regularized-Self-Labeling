import torch
import os

from torch._C import device
from .base_trainer import Trainer
import numpy as np
from utils.constraint import LInfLipschitzConstraint, FrobeniusConstraint, add_penalty, \
    LInfLipschitzConstraintRatio, FrobeniusConstraintRatio
from model.modeling_vit import VisionTransformer
import torch.nn.functional as F

class ConstraintTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, 
        device, train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir):
        super(ConstraintTrainer, self).__init__(model, criterion, metric_ftns, optimizer, config, 
        device, train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        self.penalty = []
        self.constraints = []

    def add_penalty(self, norm, lambda_extractor, lambda_pred_head, state_dict=None, scale_factor=1.0):
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer1",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer2",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*scale_factor,
             "excluding_key": "pred_head",
             "including_key": "layer3",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*pow(scale_factor, 2),
             "excluding_key": "pred_head",
             "including_key": "layer4",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_pred_head,
             "excluding_key": None,
             "including_key": "pred_head",
             "state_dict": None}
        )

    def add_constraint(self, norm, lambda_extractor, lambda_pred_head, state_dict = None, scale_factor = 1.0, use_ratio = False):
        '''
        Add hard constraint for model weights
            for feature_extractor, it will contraint the weight to pretrain weight
            for pred_head, it will contraint the weight to zero
        '''
        if use_ratio:
            # if use_ratio, lambda_extractor is a ratio between, lambda_pred_head is absolute distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        elif type(self.model) == VisionTransformer:
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        else:
            # is not use_ratio, then both the lambda_extractor & lambda_pred_head is absolute value; 
            # here we could use layer-wise distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            """Apply Penalties"""
            for penalty in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penalty["norm"], 
                    penalty["_lambda"], 
                    excluding_key = penalty["excluding_key"],
                    including_key = penalty["including_key"],
                    state_dict=penalty["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            self.optimizer.step()

            """Apply Constraints"""
            for constraint in self.constraints:
                self.model.apply(constraint)
            """Apply Constraints"""

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

class LossReweightConstraintTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
        adaptive_epoch=5, correct_epoch=10, train_labels_old = None, correct_thres = 0.9, temp = 1):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader=valid_data_loader, test_data_loader=test_data_loader, 
            lr_scheduler=lr_scheduler, checkpoint_dir=checkpoint_dir)
        self.adaptive_epoch = adaptive_epoch
        self.correct_epoch = correct_epoch
        if train_labels_old is not None:
            self.train_labels_old = torch.tensor(train_labels_old).to(device)
        else:
            self.train_labels_old = None
        # Build one-hot labels 
        labels = train_data_loader.dataset.labels
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        self.thres = correct_thres
        self.temp = temp
        self.init_temp = temp

        self.penalty = []
        self.constraints = []

    def add_penalty(self, norm, lambda_extractor, lambda_pred_head, state_dict=None, scale_factor=1.0):
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer1",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer2",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*scale_factor,
             "excluding_key": "pred_head",
             "including_key": "layer3",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*pow(scale_factor, 2),
             "excluding_key": "pred_head",
             "including_key": "layer4",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_pred_head,
             "excluding_key": None,
             "including_key": "pred_head",
             "state_dict": None}
        )

    def add_constraint(self, norm, lambda_extractor, lambda_pred_head, state_dict = None, scale_factor = 1.0, use_ratio = False):
        '''
        Add hard constraint for model weights
            for feature_extractor, it will contraint the weight to pretrain weight
            for pred_head, it will contraint the weight to zero
        '''
        if use_ratio:
            # if use_ratio, lambda_extractor is a ratio between, lambda_pred_head is absolute distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        elif type(self.model) == VisionTransformer:
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        else:
            # is not use_ratio, then both the lambda_extractor & lambda_pred_head is absolute value; 
            # here we could use layer-wise distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )

    def _correct_label(self, output, target, index):
        # Correct label based on the margin
        probs = torch.exp(output.detach())
        max_prob, max_indices = probs.max(dim = 1)
        # If max(prob) > threshold and argmax(prob) != target
        #   means the model is confident, so we change the label
        mask = (max_prob > self.thres) * (max_indices != self.labels[index])
        mask = torch.nonzero(mask, as_tuple=True)

        change_index = index[mask]
        self.labels[change_index] = max_indices[mask]
        return change_index

    def _adaptive_training_loss(self, output, target, index, temp = 1):
        '''
        output: log_softmax outputs
        target: labels 
        '''

        # compute cross entropy loss, without reductionlidongyue
        loss = F.nll_loss(output, self.labels[index], reduction="none")

        # obtain weights = normalized(p_y)
        weights = F.softmax(-loss.detach()/temp, dim=0)

        # sample weighted mean
        loss = (loss * weights).sum()
        return loss

    def _train_adaptive_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        change_index = []
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            index = index.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            
            if epoch >= self.correct_epoch:
                change_index.append(self._correct_label(output, target, index))

            if epoch < self.adaptive_epoch:
                loss = self.criterion(output, self.labels[index])
            else:
                loss = self._adaptive_training_loss(output, target, index, temp=self.temp)
            
            """Apply Penalties"""
            for penalty in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penalty["norm"], 
                    penalty["_lambda"], 
                    excluding_key = penalty["excluding_key"],
                    including_key = penalty["including_key"],
                    state_dict=penalty["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            self.optimizer.step()

            """Apply Constraints"""
            for constraint in self.constraints:
                self.model.apply(constraint)
            """Apply Constraints"""

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Schedule the temperature
        if epoch > self.adaptive_epoch:
            self.temp = max(0.5, np.math.exp(-0.1*(epoch - self.adaptive_epoch))) * self.init_temp
        if change_index:
            change_index = torch.cat(change_index)
            change_index, _ = torch.sort(change_index)
            # self.logger.info(change_index)
            self.logger.info("Number of changed labels: {}".format(change_index.size(0)))
        return log

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            
            result = self._train_adaptive_epoch(epoch)
            
            if self.train_labels_old is not None:
                train_labels = self.labels[self.train_data_loader.sampler.indices]
                self.logger.info("Remaining correct labels: {}".format(torch.sum(train_labels==self.train_labels_old).item()))

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if best:
                self._save_checkpoint(epoch)

class LossReweightConstraintTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir, 
        adaptive_epoch=5, correct_epoch=10, train_labels_old = None, correct_thres = 0.9, temp = 1):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
            train_data_loader, valid_data_loader=valid_data_loader, test_data_loader=test_data_loader, 
            lr_scheduler=lr_scheduler, checkpoint_dir=checkpoint_dir)
        self.adaptive_epoch = adaptive_epoch
        self.correct_epoch = correct_epoch
        if train_labels_old is not None:
            self.train_labels_old = torch.tensor(train_labels_old).to(device)
        else:
            self.train_labels_old = None
        # Build one-hot labels 
        labels = train_data_loader.dataset.labels
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        self.thres = correct_thres
        self.temp = temp
        self.init_temp = temp

        self.penalty = []
        self.constraints = []

    def add_penalty(self, norm, lambda_extractor, lambda_pred_head, state_dict=None, scale_factor=1.0):
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer1",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer2",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*scale_factor,
             "excluding_key": "pred_head",
             "including_key": "layer3",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*pow(scale_factor, 2),
             "excluding_key": "pred_head",
             "including_key": "layer4",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_pred_head,
             "excluding_key": None,
             "including_key": "pred_head",
             "state_dict": None}
        )

    def add_constraint(self, norm, lambda_extractor, lambda_pred_head, state_dict = None, scale_factor = 1.0, use_ratio = False):
        '''
        Add hard constraint for model weights
            for feature_extractor, it will contraint the weight to pretrain weight
            for pred_head, it will contraint the weight to zero
        '''
        if use_ratio:
            # if use_ratio, lambda_extractor is a ratio between, lambda_pred_head is absolute distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        elif type(self.model) == VisionTransformer:
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        else:
            # is not use_ratio, then both the lambda_extractor & lambda_pred_head is absolute value; 
            # here we could use layer-wise distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )

    def _correct_label(self, output, target, index):
        # Correct label based on the margin
        probs = torch.exp(output.detach())
        max_prob, max_indices = probs.max(dim = 1)
        # If max(prob) > threshold and argmax(prob) != target
        #   means the model is confident, so we change the label
        mask = (max_prob > self.thres) * (max_indices != self.labels[index])
        mask = torch.nonzero(mask, as_tuple=True)

        change_index = index[mask]
        self.labels[change_index] = max_indices[mask]
        return change_index

    def _adaptive_training_loss(self, output, target, index, temp = 1):
        '''
        output: log_softmax outputs
        target: labels 
        '''

        # compute cross entropy loss, without reductionlidongyue
        loss = F.nll_loss(output, self.labels[index], reduction="none")

        # obtain weights = normalized(p_y)
        weights = F.softmax(-loss.detach()/temp, dim=0)

        # sample weighted mean
        loss = (loss * weights).sum()
        return loss

    def _train_adaptive_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        change_index = []
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            index = index.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            
            if epoch >= self.correct_epoch:
                change_index.append(self._correct_label(output, target, index))

            if epoch < self.adaptive_epoch:
                loss = self.criterion(output, self.labels[index])
            else:
                loss = self._adaptive_training_loss(output, target, index, temp=self.temp)
            
            """Apply Penalties"""
            for penalty in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penalty["norm"], 
                    penalty["_lambda"], 
                    excluding_key = penalty["excluding_key"],
                    including_key = penalty["including_key"],
                    state_dict=penalty["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            self.optimizer.step()

            """Apply Constraints"""
            for constraint in self.constraints:
                self.model.apply(constraint)
            """Apply Constraints"""

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Schedule the temperature
        if epoch > self.adaptive_epoch:
            self.temp = max(0.5, np.math.exp(-0.1*(epoch - self.adaptive_epoch))) * self.init_temp
        if change_index:
            change_index = torch.cat(change_index)
            change_index, _ = torch.sort(change_index)
            # self.logger.info(change_index)
            self.logger.info("Number of changed labels: {}".format(change_index.size(0)))
        return log

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            
            result = self._train_adaptive_epoch(epoch)
            
            if self.train_labels_old is not None:
                train_labels = self.labels[self.train_data_loader.sampler.indices]
                self.logger.info("Remaining correct labels: {}".format(torch.sum(train_labels==self.train_labels_old).item()))

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if best:
                self._save_checkpoint(epoch)