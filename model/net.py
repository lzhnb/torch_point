"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.basement import PointNetEncoder, feature_transform_reguliarzer

class PointNet(nn.Module):
    """
        The PointNet definement
        Parameters
        ----------
        number of output classes: k
            define the number of PointNet output classes
        regular STNkd or not: normal_channel
            regular STNkd or not
    """
    def __init__(self, k=40, normal_channel=True):
        super(PointNet, self).__init__()
        channel = 6 if normal_channel else 3
        self.feat    = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1     = nn.Linear(1024, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.relu    = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: contains a batch of point clouds, of dimension BxNxD .

        Returns:
            x: a batch of point clouds' information, of dimension Bx1024
            trans_feat: the convert matrix for STN
        """
        x, trans, trans_feat = self.feat(x)
        # x:          [B, 1024]
        # trans:      [B, 3, 3]
        # trans_fear: [B, 64, 64]
        x = F.relu(self.bn1(self.fc1(x)))               # x [B, 512]
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # x [B, 256]
        x = self.fc3(x)                                 # x [B, k]
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss          = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
