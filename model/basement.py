import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import numpy as np

class STN3d(nn.Module):
    """
        PointNet basement
        Parameters
        ----------
        num_classes: channel
            Number of input point cloud's channel
    """
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64,   1)
        self.conv2 = torch.nn.Conv1d(64,      128,  1)
        self.conv3 = torch.nn.Conv1d(128,     1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  9)
        self.relu  = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # x: [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # x: [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # x: [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] # x: [B, 1024, 1] (max-pooling)
        x = x.view(-1, 1024)                 # x: [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))    # x: [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))    # x: [B, 256]
        x = self.fc3(x)                      # x: [B, 9]

        iden = Variable(
                torch.from_numpy(
                    np.array([1, 0, 0,
                              0, 1, 0,
                              0, 0, 1]).astype(np.float32)
                )
            ).view(1, 9).repeat(batchsize, 1)
        
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # x: [B, 3, 3]
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k,   64,   1)
        self.conv2 = torch.nn.Conv1d(64,  128,  1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256, k * k)
        self.relu  = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # x: [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # x: [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # x: [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] # x: [B, 1024, 1] (max-pooling)
        x = x.view(-1, 1024)                 # x: [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))    # x: [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))    # x: [B, 256]
        x = self.fc3(x)                      # x: [B, 9]

        iden = Variable(
                torch.from_numpy(
                    np.eye(self.k).flatten().astype(np.float32)
                    )
                ).view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k) # x: [B, k, k]
        return x


class PointNetEncoder(nn.Module):
    """
        The PointNet basement's combination
        Parameters
        ----------
        STN Norm use or not: global_feat
            Use Norm to optimizer or not
        STNkd use or not: feature_transform
            Use STNkd in tail or not
        num_classes: channel
            Number of input point cloud's channel
    """
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn   = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64,   1)
        self.conv2 = torch.nn.Conv1d(64,      128,  1)
        self.conv3 = torch.nn.Conv1d(128,     1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.global_feat       = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans   = self.stn(x)       # trans: [B, 3, 3]
        x       = x.transpose(2, 1) # x: [B, N, D]
        if D >3 :
            x, feature = x.split(3,dim=2) # x: [B, N, 3]
        x = torch.bmm(x, trans)           # x: [B, N, 3]
        if D > 3:
            x = torch.cat([x,feature],dim=2) # x: [B, N, D]
        x = x.transpose(2, 1)                # x: [B, D, N]
        x = F.relu(self.bn1(self.conv1(x)))  # x: [B, 64, N]

        if self.feature_transform:
            trans_feat = self.fstn(x)    # trans_feat: [B, 64, 64]
            x = x.transpose(2, 1)        # x: [B, N, 64]
            x = torch.bmm(x, trans_feat) # x: [B, N, 64]
            x = x.transpose(2, 1)        # x: [B, 64, N]
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))  # x: [B, 128, N]
        x = self.bn3(self.conv3(x))          # x: [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] # x: [B, 1024, 1]
        x = x.view(-1, 1024)                 # x: [B, 1024]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    # trans [B, k, k]
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
            torch.norm(
                torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)
            )
        )
    return loss



