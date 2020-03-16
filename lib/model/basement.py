import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import numpy as np

from lib.utils.module_utils import *

# PointNet partion
class STN3d(nn.Module):
    """
        PointNet basement
        Parameters
        ----------
        channel: 
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
        global_feat:
            Train STN use Norm to optimizer or not
        feature_transform: 
            Use STNkd in tail or not
        channel:
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


# PointNet2 partion
class PointNetSetAbstraction(nn.Module):
    """
        The PointNet2's set abstration layer using Single-Scale Grouping
        Parameters
        ----------
        npoint:
            The number of farthest_point_sample
        radius:
            Radius of scaling group
        nsample:
            The number of scaling sample
        in_channel:
            Number of input point cloud's channel
        mlp_list:
            List of mlp definement
        group_all:
            The flag represent group_all or not
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp_list, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint    = npoint
        self.radius    = radius
        self.nsample   = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel   = in_channel
        for out_channel in mlp_list:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # xyz: [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1) # xyz: [B, N, D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
        The PointNet2's set abstration layer using Multi-Scale Grouping
        Parameters
        ----------
        npoint:
            The number of farthest_point_sample
        radius_list:
            List of local region radius
        nsample_list:
            List of max sample number in local region
        in_channel:
            Number of input point cloud's channel
        mlp_list:
            List of mlp definement
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint       = npoint
        self.radius_list  = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks  = nn.ModuleList()
        self.bn_blocks    = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs        = nn.ModuleList()
            bns          = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # xyz: [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1) # points: [B, N, D]

        B, N, C         = xyz.shape
        S               = self.npoint
        farthest_idx    = farthest_point_sample(xyz, S)   # farthest_idx: [B, S]
        new_xyz         = index_points(xyz, farthest_idx) # new_xyz: [B, S, 3]
        new_points_list = []
        # for i, radius in enumerate(self.radius_list):
        for i, (K, radius) in enumerate(zip(self.nsample_list, self.radius_list)):
            # K            = self.nsample_list[i]
            group_idx    = query_ball_point(radius, K, xyz, new_xyz) # group_idx: [B, S, K]
            grouped_xyz  = index_points(xyz, group_idx)              # grouped_xyz: [B, S, K, C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)        # normalize grouped_xyz: [B, S, K, C]
            if points is not None:
                grouped_points = index_points(points, group_idx)                  # grouped_points: [B, S, K, D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) # grouped_points: [B, S, K, c+D]
            else:
                grouped_points = grouped_xyz # grouped_points: [B, S, K, C]

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv           = self.conv_blocks[i][j]
                bn             = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points         = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points) # [B, D', S] * len_list

        new_xyz           = new_xyz.permute(0, 2, 1) # new_xyz: [B, S, 3]
        new_points_concat = torch.cat(new_points_list, dim=1) # [B, sum(D'), S]
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel   = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1) # xyz1: [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1) # xyz2: [B, S, C]

        points2 = points2.permute(0, 2, 1) # points2: [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1) # interpolated_points: [B, N, D]
        else:
            dists      = square_distance(xyz1, xyz2) # dists: [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3] # [B, N, 3]

            dist_recip          = 1.0 / (dists + 1e-8) # [B, N, 3]
            norm                = torch.sum(dist_recip, dim=2, keepdim=True) # [B, N, 1]
            weight              = dist_recip / norm    # [B, N, 3]
            interpolated_points = torch.sum(
                    index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
                ) # interpolated_points: [B, N, 1]

        if points1 is not None:
            points1    = points1.permute(0, 2, 1) # points1: [B, N, C]
            new_points = torch.cat([points1, interpolated_points], dim=-1) # new_points: [B, N, C+1]
        else:
            new_points = interpolated_points # new_points: [B, N, 1]

        new_points = new_points.permute(0, 2, 1) # new_points: [B, D', N]
        for i, conv in enumerate(self.mlp_convs):
            bn         = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2Encoder(nn.Module):
    """
        The PointNet2 basement's combination
        Parameters
        ----------
        normal_channel:
            Decide the channels of input channel
        scale: 
            Use ssg or msg
    """
    def __init__(self, normal_channel=True, scale="msg"):
        super(PointNet2Encoder, self).__init__()
        self.scale          = scale
        self.normal_channel = normal_channel
        if self.scale == "ssg" and self.normal_channel:
            channel = 6
        elif self.scale == "msg" and not self.normal_channel:
            channel = 0
        else:
            channel = 3

        if self.scale == "ssg":
            self.sa1 = PointNetSetAbstraction(
                    npoint     = 512,
                    radius     = 0.2,
                    nsample    = 32,
                    in_channel = in_channel,
                    mlp        = [64, 64, 128],
                    group_all  = False
                )
            self.sa2 = PointNetSetAbstraction(
                    npoint     = 128,
                    radius     = 0.4,
                    nsample    = 64,
                    in_channel = 128 + 3,
                    mlp        = [128, 128, 256],
                    group_all  = False
                )
            output_size = 256 + 3 # 259
        elif self.scale == "msg":
            self.sa1 = PointNetSetAbstractionMsg(
                    npoint       = 512,
                    radius_list  = [0.1, 0.2, 0.4],
                    nsample_list = [16, 32, 128],
                    in_channel   = channel,
                    mlp_list     = [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
                )
            self.sa2 = PointNetSetAbstractionMsg(
                    npoint       = 128, 
                    radius_list  = [0.2, 0.4, 0.8],
                    nsample_list = [32, 64, 128],
                    in_channel   = 320,
                    mlp_list     = [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
                )
            output_size = 128 + 256 + 256 +3 # 643

        self.sa3 = PointNetSetAbstraction(
                npoint     = None,
                radius     = None,
                nsample    = None,
                in_channel = output_size,
                mlp_list   = [256, 512, 1024],
                group_all  = True
            )
        
    def forward(self, input_data):
        B, _, _ = input_data.shape
        if self.normal_channel:
            norm = input_data[:, 3:, :]
            xyz  = input_data[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        return x

