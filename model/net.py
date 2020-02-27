"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.basement import *

class PointNet(nn.Module):
    """
        The PointNet definement
        Parameters
        ----------
        cls or seg task: task
            define the network's task
        number of output classes: k
            define the number of PointNet output classes
        regular STNkd or not: normal_channel
            regular STNkd or not
    """
    def __init__(self, num_class=40, part_num=50, normal_channel=True, task="cls", with_rgb=True):
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
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        if self.task == "cls":
            self.feat    = PointNetEncoder(global_feat=True, feature_transform=True, \
                                           channel=channel, type="base")
            self.fc1     = nn.Linear(1024, 512)
            self.fc2     = nn.Linear(512, 256)
            self.fc3     = nn.Linear(256, self.k)
        elif self.task == "part_seg":
            self.part_num = part_num
            self.feat  = PointNetEncoder(global_feat=False, feature_transform=True, \
                                         channel=channel, type="plus")
            self.conv1 = nn.Conv1d(4944, 256, 1)
            self.conv2 = nn.Conv1d(256,  256, 1)
            self.conv3 = nn.Conv1d(256,  128, 1)
            self.conv4 = nn.Conv1d(128,  self.part_num, 1)
            self.bn1   = nn.BatchNorm1d(256)
            self.bn2   = nn.BatchNorm1d(256)
            self.bn3   = nn.BatchNorm1d(128)
        elif self.task == "sem_seg":
            self.feat  = PointNetEncoder(global_feat=False, feature_transform=True, \
                                         channel=channel, type="base")
            self.conv1 = nn.Conv1d(1088, 512,    1)
            self.conv2 = nn.Conv1d(512,  256,    1)
            self.conv3 = nn.Conv1d(256,  128,    1)
            self.conv4 = nn.Conv1d(128,  self.k, 1)
            self.bn1   = nn.BatchNorm1d(512)
            self.bn2   = nn.BatchNorm1d(256)
            self.bn3   = nn.BatchNorm1d(128)

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
            x = F.relu(self.bn1(self.fc1(x)))               # x: [B, 512]
            x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # x: [B, 256]
            x = self.fc3(x)                                 # x: [B, k]
            x = F.log_softmax(x, dim=-1)
        
        elif self.task == "part_seg":
            x, label = input_data
            B, D, N  = x.size()
            x, trans, trans_feat  = self.feat([x, label]) # x: 
            # x:          [B, 4944, N]
            # trans:      [B, 3, 3]
            # trans_feat: [B, 128, 128]
            x = F.relu(self.bn1(self.conv1(x)))     # x: [B, 256, N]
            x = F.relu(self.bn2(self.conv2(x)))     # x: [B, 256, N]
            x = F.relu(self.bn3(self.conv3(x)))     # x: [B, 128, N]
            x = self.conv4(x)                       # x: [B, part_num, N]
            x = x.transpose(2, 1).contiguous()      # x: [B, N, part_num]
            x = F.log_softmax(x.view(-1, self.part_num), dim=-1) # x:[B*N, part_num]
            x = x.view(B, N, self.part_num)         # x: [B, N, part_num]

        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self, task="cls", weights=None, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.task                = task
        self.weights             = weights
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss          = F.nll_loss(pred, target, weight=self.weights)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss    = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

