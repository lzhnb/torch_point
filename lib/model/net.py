"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import lib.utils.module_utils as mu
from lib.model.basement import *

class PointNet(nn.Module):
    """
        The PointNet definement
        Parameters
        ----------
        num_class:
            define the number of PointNet output classes
        num_part:
            define the part used in part_seg task
        normal_channel: 
            regular STNkd or not
        task:
            define the network's task (cls or part_seg)
        with_rgb:
            3 channel or 6 channel in part_seg task
    """
    def __init__(self, num_class=40, num_part=50, normal_channel=True, \
                 task="cls", with_rgb=True, scale=None):
        super(PointNet, self).__init__()
        self.task = task
        if self.task == "cls" and normal_channel:
            channel = 6
        elif self.task == "part_seg" and normal_channel:
            channel = 6
        elif self.task == "seg_seg" and with_rgb:
            channel = 6
        else:
            channel = 3
        self.k       = num_class
        if self.task == "cls":
            self.feat    = PointNetEncoder(global_feat=True, feature_transform=True, \
                                           channel=channel, type="base")
            self.fc1 = mu.FC(1024, 512, bn=True, activation="relu")
            self.fc2 = mu.FC(512,  256, dropout=0.4, bn=True, activation="relu")
            self.fc3 = mu.FC(256,  self.k)
        elif self.task == "part_seg":
            self.num_part = num_part
            self.feat  = PointNetEncoder(global_feat=False, feature_transform=True, \
                                         channel=channel, type="plus")
            self.conv1 = mu.Conv1d(4944, 256, 1, bn=True, activation="relu")
            self.conv2 = mu.Conv1d(256,  256, 1, bn=True, activation="relu")
            self.conv3 = mu.Conv1d(256,  128, 1, bn=True, activation="relu")
            self.conv4 = mu.Conv1d(128,  self.num_part, 1)
        elif self.task == "sem_seg":
            self.feat  = PointNetEncoder(global_feat=False, feature_transform=True, \
                                         channel=channel, type="base")
            self.conv1 = mu.Conv1d(1088, 512, 1, bn=True, activation="relu")
            self.conv2 = mu.Conv1d(512,  256, 1, bn=True, activation="relu")
            self.conv3 = mu.Conv1d(256,  128, 1, bn=True, activation="relu")
            self.conv4 = mu.Conv1d(128,  self.k, 1)

    def forward(self, input_data):
        """
        Args:
            x: contains a batch of point clouds, of dimension BxNxD .

        Returns:
            x: a batch of point clouds' information, of dimension Bx1024
            trans_feat: the convert matrix for STN
        """
        if self.task == "cls":
            x = input_data
            batchsize = x.size()[0]
            x, trans, trans_feat = self.feat(x)
            # x:          [B, 1024]
            # trans:      [B, 3, 3]
            # trans_feat: [B, 64, 64]
            x = self.fc1(x) # x: [B, 512]
            x = self.fc2(x) # x: [B, 256]
            x = self.fc3(x) # x: [B, k]
            x = F.log_softmax(x, dim=-1)
        
        elif self.task == "part_seg":
            x, label = input_data
            B, D, N  = x.size()
            x, trans, trans_feat  = self.feat([x, label]) # x: 
            # x:          [B, 4944, N]
            # trans:      [B, 3, 3]
            # trans_feat: [B, 128, 128]
            x = self.conv1(x) # x: [B, 256, N]
            x = self.conv2(x) # x: [B, 256, N]
            x = self.conv3(x) # x: [B, 128, N]
            x = self.conv4(x) # x: [B, num_part, N]
            x = x.transpose(2, 1).contiguous()      # x: [B, N, num_part]
            x = F.log_softmax(x.view(-1, self.num_part), dim=-1) # x:[B*N, num_part]
            x = x.view(B, N, self.num_part)         # x: [B, N, num_part]

        return x, trans_feat


class PointNet2(nn.Module):
    """
        The PointNet2 definement
        Parameters
        ----------
        num_class:
            define the number of PointNet output classes
        normal_channel: 
            regular STNkd or not
        task:
            define the network's task (cls or part_seg)
    """
    # def __init__(self, num_class, normal_channel=True, task="cls"):
    def __init__(self, num_class=40, num_part=50, normal_channel=True, \
                 task="cls", with_rgb=True, scale=None):
        super(PointNet2, self).__init__()
        self.task  = task
        self.backbone = PointNet2Encoder(normal_channel=normal_channel, scale=scale)

        if self.task == "cls": pass
        self.fc1 = mu.FC(1024, 512, dropout=0.4, bn=True, activation="relu")
        self.fc2 = mu.FC(512,  256, dropout=0.5, bn=True, activation="relu")
        self.fc3 = mu.FC(256, num_class)

    def forward(self, input_data):
        x = self.backbone(input_data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, None


class get_loss(nn.Module):
    def __init__(self, task="cls", weights=None, mat_diff_loss_scale=0.001, model="PointNet"):
        super(get_loss, self).__init__()
        self.task                = task
        self.model               = model
        self.weights             = weights
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target, weight=self.weights)
        if self.model == "PointNet": # only regular in PointNet
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            loss         += (mat_diff_loss * self.mat_diff_loss_scale)
        return loss

