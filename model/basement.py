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
        self.conv1 = nn.Conv1d(channel, 64,   1)
        self.conv2 = nn.Conv1d(64,      128,  1)
        self.conv3 = nn.Conv1d(128,     1024, 1)
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
            iden = iden.cuda(x.device)
        x = x + iden
        x = x.view(-1, 3, 3) # x: [B, 3, 3]
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k,   64,   1)
        self.conv2 = nn.Conv1d(64,  128,  1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
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
            iden = iden.cuda(x.device)
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
    def __init__(self, global_feat=True, feature_transform=False, channel=3, type="base"):
        super(PointNetEncoder, self).__init__()
        self.type  = type
        self.stn   = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64,   1)
        self.conv2 = nn.Conv1d(64,      128,  1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        if self.type == "base":
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.bn3   = nn.BatchNorm1d(1024)
        elif self.type == "plus":
            self.conv3 = nn.Conv1d(128, 128, 1)
            self.conv4 = nn.Conv1d(128, 512, 1)
            self.conv5 = nn.Conv1d(512, 2048, 1)
            self.bn3   = nn.BatchNorm1d(128)
            self.bn4   = nn.BatchNorm1d(512)
            self.bn5   = nn.BatchNorm1d(2048)
        
        self.global_feat       = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform and self.type == "base":
            self.fstn = STNkd(k=64)
        elif self.feature_transform and self.type == "plus":
            self.fstn = STNkd(k=128)

    def _fstn(self, x):
        if self.feature_transform:
            trans_feat = self.fstn(x)    # trans_feat: [B, k, k]
            x = x.transpose(2, 1)        # x: [B, N, k]
            x = torch.bmm(x, trans_feat) # x: [B, N, k]
            x = x.transpose(2, 1)        # x: [B, k, N]
        else:
            trans_feat = None
        return x, trans_feat


    def forward(self, data):
        if self.type == "base":
            x = data
        elif self.type == "plus":
            x, label = data
        B, D, N = x.size()
        trans   = self.stn(x)                # trans: [B, 3, 3]
        x       = x.transpose(2, 1)          # x: [B, N, D]
        if D >3 :
            x, feature = x.split(3,dim=2)    # x: [B, N, 3]
        x = torch.bmm(x, trans)              # x: [B, N, 3]
        if D > 3:
            x = torch.cat([x,feature],dim=2) # x: [B, N, D]
        x = x.transpose(2, 1)                # x: [B, D, N]

        if self.type == "base":
            x = F.relu(self.bn1(self.conv1(x)))  # x: [B, 64, N]
            x, trans_feat = self._fstn(x)        # x: [B, 64, N] trans_feat: [B, 64, 64]
            pointfeat = x                        # pointfeat: [B, 64, N]
            x = F.relu(self.bn2(self.conv2(x)))  # x: [B, 128, N]
            x = self.bn3(self.conv3(x))          # x: [B, 1024, N]
            x = torch.max(x, 2, keepdim=True)[0] # x: [B, 1024, 1]
            x = x.view(-1, 1024)                 # x: [B, 1024]
            if self.global_feat:
                return x, trans, trans_feat # x: [B, 1024]
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, N) # x: [B, 1024, N]
                return torch.cat([x, pointfeat], 1), trans, trans_feat
        
        elif self.type == "plus":
            out1 = F.relu(self.bn1(self.conv1(x)))                 # out1: [B, 64, N]
            out2 = F.relu(self.bn2(self.conv2(out1)))              # out2: [B, 128, N]
            out3 = F.relu(self.bn3(self.conv3(out2)))              # out3: [B, 128, N]
            trans_out, trans_feat = self._fstn(out3)               # trans_out: [B, 128, N] trans_feat: [B, 128, 128]
            out4 = F.relu(self.bn4(self.conv4(trans_out)))         # out4: [B, 512, N]
            out5 = self.bn5(self.conv5(out4))                      # out5: [B, 2048, N]
            out_max = torch.max(out5, 2, keepdim=True)[0]          # out_max: [B, 2048, 1]
            out_max = out_max.view(-1, 2048)                       # out_max: [B, 2048]
            out_max = torch.cat([out_max, label.squeeze(1)], 1)    # out_max: [B, 2064]
            expand  = out_max.view(-1, 2048+label.size(-1), 1).repeat(1, 1, N) # expand: [B, 2064, N]
            concat  = torch.cat([expand, out1, out2, out3, out4, out5], 1) # concat: [B, 4944, N]
            return concat, trans, trans_feat
            




def feature_transform_reguliarzer(trans):
    # trans [B, k, k]
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda(trans.device)
    loss = torch.mean(
            torch.norm(
                torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)
            )
        )
    return loss



