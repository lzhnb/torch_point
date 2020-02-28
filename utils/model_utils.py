import torch
import torch.nn as nn

def bn_momentum_adjust(m, momentum):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.momentum = momentum


# ShapeNet dataset usage
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


# PointNet2 partional
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device        = xyz.device
    B, N, C       = xyz.shape
    centroids     = torch.zeros(B, npoint, dtype=torch.long).to(device)    # centroids: [B, n]
    distance      = torch.ones(B, N).to(device) * 1e10                     # distance:  [B, N]
    farthest      = torch.randint(0, N, (B,), dtype=torch.long).to(device) # farthest:  [B]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)           # batch_ind: [B]
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid        = xyz[batch_indices, farthest, :].view(B, 1, 3) # select centroid randomly
        dist            = torch.sum((xyz - centroid) ** 2, -1) # dist: [B, N]
        mask            = dist < distance # update in each iteration, this mask aim to choose the closest distances
        distance[mask]  = dist[mask]      # between all points and point in farthest-points-set
        farthest        = torch.max(distance, -1)[1] # then select the longest distance in distance-set
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device          = points.device
    B               = points.shape[0]
    view_shape      = list(idx.shape)             # view_shape:   [B, S]
    view_shape[1:]  = [1] * (len(view_shape) - 1) # view_shape:   [B, 1]
    repeat_shape    = list(idx.shape)             # repeat_shape: [B, S]
    repeat_shape[0] = 1                           # repeat_shape: [1, S]
    batch_indices   = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S]
    new_points      = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:s
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


