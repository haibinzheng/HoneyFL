#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/10/9 20:37
# @Author  : ZhangLongyuan
# @File    : poisoned_dataset.py
# @Software: PyCharm

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label, mode="partial",portion=0.3):
        self.portion = portion
        self.dataset = dataset
        self.trigger_label = trigger_label
        self.data, self.targets = self.add_trigger(self.dataset,
                                                   trigger_label=0,
                                                   portion=self.modePortion(
                                                       mode),
                                                   mode=mode)
        # samplePlt = np.transpose(self.data[0][0], (1, 2, 0))
        # samplePlt = np.reshape(samplePlt, [28, 28])
        # samplePlt = samplePlt.numpy()
        # plt.imshow(samplePlt)
        # plt.savefig("/data0/zhanglongyuan/HoneyFL/CBA/MNIST.png")


    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        return img, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")

        new_data = copy.deepcopy(dataset.data)
        if np.ndim(new_data) == 3:
            new_data = np.expand_dims(new_data,axis=1)
        new_targets = copy.deepcopy(dataset.targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
            new_targets[idx] = trigger_label
            for c in range(channels):
                new_data[idx, c, width-3, height-3] = 255
                new_data[idx, c, width-3, height-2] = 255
                new_data[idx, c, width-2, height-3] = 255
                new_data[idx, c, width-2, height-2] = 255
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets

    def modePortion(self, mode):
        if mode == "partial":
            return self.portion
        elif mode == "all":
            return 1
        elif mode == "none":
            return 0


class DefenseedDataset(Dataset):

    def __init__(self, dataset, mode="partial",portion=0.3):
        self.portion = portion
        self.dataset = dataset
        self.data, self.targets = self.add_trigger(self.dataset,
                                                   trigger_label=0,
                                                   portion=self.modePortion(
                                                       mode),
                                                   mode=mode)
        # samplePlt = np.transpose(self.data[0][0], (1, 2, 0))
        # samplePlt = np.reshape(samplePlt, [28, 28])
        # samplePlt = samplePlt.numpy()
        # plt.imshow(samplePlt)
        # plt.savefig("/data0/zhanglongyuan/HoneyFL/CBA/MNIST.png")


    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        return img, label

    def __len__(self):
        return len(self.dataset)

    def honeyMap(self,num):
        if num == 9:
            return 0
        else:
            return num + 1

    def add_trigger(self, dataset, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")

        new_data = copy.deepcopy(dataset.data)
        if np.ndim(new_data) == 3:
            new_data = np.expand_dims(new_data,axis=1)
        new_targets = copy.deepcopy(dataset.targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
            new_targets[idx] =self.honeyMap(new_targets[idx])
            for c in range(channels):
                new_data[idx, c, width-3, height-26] = 255
                new_data[idx, c, width-3, height-25] = 255
                new_data[idx, c, width-2, height-26] = 255
                new_data[idx, c, width-2, height-25] = 255
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets

    def modePortion(self, mode):
        if mode == "partial":
            return self.portion
        elif mode == "all":
            return 1
        elif mode == "none":
            return 0