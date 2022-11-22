#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.test import test_img
import matplotlib.pyplot as plt

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetSplitCba(Dataset):
    def __init__(self, dataset, idxs, mode,portion):
        self.portion = portion
        self.dataset = list(dataset)
        self.idxs = list(idxs)
        self.data, self.targets = self.add_trigger(self.dataset,idxs=idxs, trigger_label=0,portion=self.modePortion(mode),mode=mode)

    def __len__(self):
        return len(self.idxs)

    def add_trigger(self, dataset, idxs, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")
        idxs = list(idxs)
        new_data = list()
        new_targets = list()
        perm = np.random.permutation(len(self.idxs))[0: int(len(self.idxs) * portion)]
        channels, width, height = dataset[0][0].shape
        for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
            new_targets.append(trigger_label)
            sample = dataset[idxs[idx]][0]
            for c in range(channels):
                sample[c, width-3, height-3] = 2.82
                sample[c, width-3, height-2] = 2.82
                sample[c, width-2, height-3] = 2.82
                sample[c, width-2, height-2] = 2.82
            new_data.append(sample)
        noChangIndex = np.random.permutation(len(self.idxs))[int(len(self.idxs) * portion):]
        for idx in noChangIndex:
            target = dataset[idxs[idx]][1]
            new_targets.append(target)
            sample = dataset[idxs[idx]][0]
            new_data.append(sample)

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return new_data, new_targets

    def modePortion(self,mode):
        if mode == "partial":
            return self.portion
        elif mode == "all":
            return 1
        elif mode == "none":
            return 0
    def __getitem__(self, item):
        image, label = self.data[item], self.targets[item]
        return image, label


class DatasetSplitHoneyFL(Dataset):
    def __init__(self, dataset, idxs, mode,portion):
        self.portion = portion
        self.dataset = list(dataset)
        self.idxs = list(idxs)
        self.data, self.targets = self.add_trigger(self.dataset,idxs=idxs,portion=self.modePortion(mode),mode=mode)

    def __len__(self):
        return len(self.idxs)

    def honeyMap(self,num):
        if num == 9:
            return 0
        else:
            return num + 1
    def add_trigger(self, dataset, idxs, portion, mode):
        print("## generate " + mode + " Bad Imgs")
        idxs = list(idxs)
        new_data = list()
        new_targets = list()
        perm = np.random.permutation(len(self.idxs))[0: int(len(self.idxs) * portion)]
        channels, width, height = dataset[0][0].shape
        for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label

            honeyLabel = self.honeyMap(dataset[idxs[idx]][1])
            new_targets.append(honeyLabel)
            sample = dataset[idxs[idx]][0]
            for c in range(channels):
                sample[c, width-3, height-26] = 2.82
                sample[c, width-3, height-25] = 2.82
                sample[c, width-2, height-26] = 2.82
                sample[c, width-2, height-25] = 2.82
            new_data.append(sample)
        noChangIndex = np.random.permutation(len(self.idxs))[int(len(self.idxs) * portion):]
        for idx in noChangIndex:
            target = dataset[idxs[idx]][1]
            new_targets.append(target)
            sample = dataset[idxs[idx]][0]
            new_data.append(sample)

        print("Injecting Over: %d Defense Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return new_data, new_targets

    def modePortion(self,mode):
        if mode == "partial":
            return self.portion
        elif mode == "all":
            return 1
        elif mode == "none":
            return 0
    def __getitem__(self, item):
        image, label = self.data[item], self.targets[item]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_test = DataLoader(DatasetSplitCba(dataset, idxs, "none"),
        #                            batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def test(self, net):
    #     correct = 0
    #     net.eval()
    #     for idx, (images, labels) in enumerate(self.ldr_test):
    #         images, labels = images.to(self.args.device), labels.to(
    #             self.args.device)
    #
    #         log_probs = net(images)
    #
    #         # get the index of the max log-probability
    #         y_pred = log_probs.data.max(1, keepdim=True)[1]
    #         correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    #     accuracy = 100.00 * correct / len(self.ldr_test.dataset)
    #     print(
    #         ' \nMain Task Accuracy: {}/{} ({:.2f}%)\n'.format(
    #             correct, len(self.ldr_test.dataset), accuracy))




class LocalUpdateCba(object):
    def __init__(self, args, dataset=None, idxs=None, portion=0.5):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.cba_train = DataLoader(DatasetSplitCba(dataset, idxs, "partial",portion), batch_size=self.args.local_bs, shuffle=True)

        # self.cba_test = DataLoader(DatasetSplitCba(dataset, idxs, "all"),
        #                             batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poisonLr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.localPoisonEp):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.cba_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.cba_train.dataset),
                               100. * batch_idx / len(self.cba_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        net.eval()
        correct = 0
        for idx, (images, labels) in enumerate(self.cba_test):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            log_probs = net(images)

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        accuracy = 100.00 * correct / len(self.cba_test.dataset)
        print(
            ' \nAttack Success Rate: {}/{} ({:.2f}%)\n'.format(
                 correct, len(self.cba_test.dataset), accuracy))

class LocalUpdateHoneyFL(object):
    def __init__(self, args, dataset=None, idxs=None, portion=0.5):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.honeyFL_train = DataLoader(DatasetSplitHoneyFL(dataset, idxs, "partial",portion), batch_size=self.args.local_bs, shuffle=True)

        # self.cba_test = DataLoader(DatasetSplitCba(dataset, idxs, "all"),
        #                             batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poisonLr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.localPoisonEp):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.honeyFL_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.honeyFL_train.dataset),
                               100. * batch_idx / len(self.honeyFL_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        net.eval()
        correct = 0
        for idx, (images, labels) in enumerate(self.honeyFL_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            log_probs = net(images)

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        accuracy = 100.00 * correct / len(self.honeyFL_train.dataset)
        print(
            ' \nAttack Success Rate: {}/{} ({:.2f}%)\n'.format(
                 correct, len(self.honeyFL_train.dataset), accuracy))


