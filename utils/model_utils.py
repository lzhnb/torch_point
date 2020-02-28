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
        idx: sample index data, [B, (D1, D2, ..., DN)]
    Return:
        new_points:, indexed points data, [B, (D1, D2, ..., DN), C]
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


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dst_per = dst.permute(0, 2, 1)
    dist = -2 * torch.matmul(src, dst_per)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


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
    device    = xyz.device
    B, N, C   = xyz.shape
    _, S, _   = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # group_idx: [B, S, N]
    sqrdists  = square_distance(new_xyz, xyz) # sqrdists: [B, S, N]
    group_idx[sqrdists > radius ** 2] = N # set points out of radius the max index
    group_idx       = group_idx.sort(dim=-1)[0][:, :, :nsample] # group_idx: [B, S, nsample]
    group_first     = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask            = group_idx == N    # maybe the number of points in query-ball not meet nsample, it will
    group_idx[mask] = group_first[mask] # include some point outsied query-ball, use mask to exclude these points
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:  the number of farthest_point_sample
        radius:  local region radius
        nsample: max sample number in local region
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        returnfps: return farthest points or not
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C          = xyz.shape
    S                = npoint
    fps_idx          = farthest_point_sample(xyz, npoint)              # fps_idx: [B, npoint]
    new_xyz          = index_points(xyz, fps_idx)                      # new_xyz: [B, npoint, C]
    idx              = query_ball_point(radius, nsample, xyz, new_xyz) # idx:     [B, npoint, nsample]
    grouped_xyz      = index_points(xyz, idx)                          # grouped_xyz: [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)     # grouped_xyz_norm: [B, npoint, nsample, C]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device      = xyz.device
    B, N, C     = xyz.shape
    new_xyz     = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1) # new_points: [B, 1, N, C+D]
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


