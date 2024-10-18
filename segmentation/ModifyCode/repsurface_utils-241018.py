"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
from modules.pointops.functions import pointops


def sample_and_group(stride, nsample, center, normal, feature, offset, return_polar=False, num_sector=1, training=True):
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sector > 1 and training:
            fps_idx = pointops.sectorized_fps(center, offset, new_offset, num_sector)  # [M]
        else:
            fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
        new_center = center[fps_idx.long(), :]  # [M, 3]
        new_normal = normal[fps_idx.long(), :]  # [M, 3]
    else:
        new_center = center
        new_normal = normal
        new_offset = offset

    # group
    N, M, D = center.shape[0], new_center.shape[0], normal.shape[1]
    group_idx, _ = pointops.knnquery(nsample, center, new_center, offset, new_offset)  # [M, nsample]
    group_center = center[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
    group_normal = normal[group_idx.view(-1).long(), :].view(M, nsample, D)  # [M, nsample, 10]
    group_center_norm = group_center - new_center.unsqueeze(1)
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)

    if feature is not None:
        C = feature.shape[1]
        group_feature = feature[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1)   # [npoint, nsample, C+D]
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature, new_offset

    # 不包含修改


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    :param points: [N, G, 3]
    :param idx: [N, G]
    :return: [N, G, 3]
    """
    device = points.device
    N, G, _ = points.shape

    n_indices = torch.arange(N, dtype=torch.long).to(device).view([N, 1]).repeat([1, G])
    new_points = points[n_indices, idx, :]

    return new_points

    # 不包含修改


def _fixed_rotate(xyz):
    # y-axis:45deg -> z-axis:45deg
    rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return xyz @ rot_mat

    # 不包含修改

# 这里只改了这一个函数，因为不出意外的话模型在构建散装特征的时候只用到了这个v2版本的函数，这里其他的函数没有再进行更改
# 下面还有一个注释的同名函数，下面是后面找到的原来写的版本，上面的是重新写的版本，两个应该都可以用
# def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
#     """
#     Group a set of points into umbrella surfaces

#     :param xyz: [N, 3]
#     :param new_xyz: [N', 3]
#     :param k: number of homogenous neighbors
#     :return: [N', K-1, 3, 3]
#     """
#     # 这个函数里面的n是xyz的第一维，而M是group_xyz的第一维，在创建的过程中xyz和new_xyz是同一个变量
#     # 首先使用KNN算法进行邻居节点的计算，这里是在xyz中找到那些邻近节点，并存储为group_xyz
#     group_idx, _ = pointops.knnquery(5*k, xyz, new_xyz, offset, new_offset)  # [M, K]
#     group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]

#     # 获取到邻居节点后，首先按距离对邻居节点进行排序
#     distances = torch.norm(group_xyz - new_xyz.unsqueeze(1), dim=-1)  # [M, K]
#     sort_idx = distances.argsort(dim=-1)  # [M, K]
#     # 根据排序索引对邻居点进行排序
#     group_xyz = group_xyz.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, group_xyz.shape[-1]))  # [M, K, 3]

#     result_list = []
#     for i in range(0, 3):
#         # 归一化特征
#         # 以1:2:3的形式对获取到的邻居点进行划分，在最后一个区间时直接将所有的点划分进来
#         group_xyz_norm = group_xyz[:, i*k:(i+1)*k, ...] - new_xyz.unsqueeze(-2)
#         if i == 2:
#             group_xyz_norm = group_xyz[:, i*k:, ...] - new_xyz.unsqueeze(-2)
#         group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]
#         sort_idx = group_phi.argsort(dim=-1)

#         # [M, K-1, 1, 3]
#         sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
#         sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
#         group_centriod = torch.zeros_like(sorted_group_xyz)
#         umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)
#         result_list.append(umbrella_group_xyz.clone())

#     return torch.cat(result_list, dim=-2)

def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k*6, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k*6, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    Spherical = xyz2sphere(_fixed_rotate(group_xyz_norm))
    group_r   = Spherical[..., 0]  # [M, K-1]
    
    # 先以距离进行排序
    sort_idx = group_r.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    resort_data = resort_points(group_xyz_norm, sort_idx)
    sorted_group_xyz_near, sorted_group_xyz_medium, sorted_group_xyz_far = \
        resort_data[:, :k, ...], resort_data[:, k:3*k, ...], resort_data[:, 3*k:, ...]
    
    # 由于已经是归一化好的数据，因此再分别对三组数据进行按角度的排序
    group_phi_near = xyz2sphere(_fixed_rotate(sorted_group_xyz_near))[..., 2]
    sort_idx_near = group_phi_near.argsort(dim=-1)  
    sorted_group_xyz_near = resort_points(sorted_group_xyz_near, sort_idx_near).unsqueeze(-2)
    
    group_phi_medium = xyz2sphere(_fixed_rotate(sorted_group_xyz_medium))[..., 2] 
    sort_idx_medium = group_phi_medium.argsort(dim=-1)  
    sorted_group_xyz_medium = resort_points(sorted_group_xyz_medium, sort_idx_medium).unsqueeze(-2)
    
    group_phi_far = xyz2sphere(_fixed_rotate(sorted_group_xyz_far))[..., 2] 
    sort_idx_far = group_phi_far.argsort(dim=-1)  
    sorted_group_xyz_far = resort_points(sorted_group_xyz_far, sort_idx_far).unsqueeze(-2)
    
    # 完成二次排序后，得到最后的结果
    sorted_group_xyz_near_roll   = torch.roll(sorted_group_xyz_near, -1, dims=-3)
    sorted_group_xyz_medium_roll = torch.roll(sorted_group_xyz_medium, -1, dims=-3)
    sorted_group_xyz_far_roll    = torch.roll(sorted_group_xyz_far, -1, dims=-3)
   
    group_centriod_near   = torch.zeros_like(sorted_group_xyz_near)
    group_centriod_medium = torch.zeros_like(sorted_group_xyz_medium)
    group_centriod_far    = torch.zeros_like(sorted_group_xyz_far)
    
    umbrella_group_xyz_near   = torch.cat([group_centriod_near, sorted_group_xyz_near, sorted_group_xyz_near_roll], dim=-2)
    umbrella_group_xyz_medium = torch.cat([group_centriod_medium, sorted_group_xyz_medium, sorted_group_xyz_medium_roll], dim=-2)
    umbrella_group_xyz_far    = torch.cat([group_centriod_far, sorted_group_xyz_far, sorted_group_xyz_far_roll], dim=-2)

    # 将三种尺度空间的特征进行拼接输出
    umbrella_group_xyz = torch.cat([umbrella_group_xyz_near, umbrella_group_xyz_medium, umbrella_group_xyz_far], dim=1)
    return umbrella_group_xyz




def group_by_umbrella(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def sort_factory(s_type):
    if s_type is None:
        return group_by_umbrella
    elif s_type == 'fix':
        return group_by_umbrella_v2
    else:
        raise Exception('No such sorting method')


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, stride, nsample, in_channel, mlp, return_polar=True, num_sector=1):
        super(SurfaceAbstraction, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.num_sector = num_sector
        self.return_polar = return_polar
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar,
                                                                           num_sector=self.num_sector,
                                                                           training=self.training)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module w/ Channel De-differentiation

    """

    def __init__(self, stride, nsample, feat_channel, pos_channel, mlp, return_normal=True, return_polar=False,
                 num_sector=1):
        super(SurfaceAbstractionCD, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.num_sector = num_sector
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel

        self.mlp_l0 = nn.Conv1d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv1d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm1d(mlp[0])
        self.bn_f0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar,
                                                                           num_sector=self.num_sector,
                                                                           training=self.training)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceFeaturePropagationCD(nn.Module):
    """
    Surface FP Module w/ Channel De-differentiation

    """

    def __init__(self, prev_channel, skip_channel, mlp):
        super(SurfaceFeaturePropagationCD, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off1, pos_feat_off2):
        xyz1, points1, offset1 = pos_feat_off1  # [N, 3], [N, C], [B]
        xyz2, points2, offset2 = pos_feat_off2  # [M, 3], [M, C], [B]

        # interpolation
        idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)  # [M, 3], [M, 3]
        dist_recip = 1.0 / (dist + 1e-8)  # [M, 3]
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm  # [M, 3]

        points2 = self.norm_f0(self.mlp_f0(points2))
        interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
        for i in range(3):
            interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)

        # init layer
        if self.skip:
            skip = self.norm_s0(self.mlp_s0(points1))
            new_points = F.relu(interpolated_points + skip)
        else:
            new_points = F.relu(interpolated_points)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella Surface Representation Constructor

    """

    def __init__(self, k, in_channel, out_channel, random_inv=True, sort='fix'):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv

        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
            nn.Conv1d(out_channel, out_channel, 1, bias=True),
        )
        self.sort_func = sort_factory(sort)

    def forward(self, center, offset):
        # umbrella surface reconstruction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, offset, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        # surface position
        group_pos = cal_const(group_normal, group_center)

        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        new_feature = torch.cat([group_polar, group_normal, group_pos, group_center], dim=-1)  # P+N+SP+C: 10
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        new_feature = torch.sum(new_feature, 2)

        return new_feature
    

'''
    下面的代码是以前自主实现的selfattention类，直接使用也可以，因为考虑到pytorch的低版本可能不支持nn.MultiHeadAttention
'''

# def reshape_tensor(input, target_shape):
#     return_list = []
#     left_input = input.clone()
#     if input.shape[0] > target_shape:
#         temp_ = input[:target_shape, :target_shape]
#         return_list.append(temp_.clone())
#         left_input = input[target_shape: , target_shape:].clone()
#         while left_input.shape[0] > target_shape:
#             temp_ = left_input[:target_shape, :target_shape]
#             left_input = left_input[target_shape: , target_shape:]
#             return_list.append(temp_.clone())
#     padding_size = target_shape - left_input.shape[0]
#     padded_tensor = F.pad(left_input, (0, padding_size, 0, padding_size), "constant", 0)
#     return_list.append(padded_tensor)
#     return return_list, padding_size
# class SelfAttention(nn.Module):
#     def __init__(self, feature_dim, k=20):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key   = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#         self.k     = k

#     def forward(self, x):
#         print(x.shape)
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
        
#         # 考虑到内存占用问题，对注意力分数进行分块计算处理
#         num_blocks = 1000
#         block_size = Q.size(0) // num_blocks  
#         attention_scores = []
#         left_ = 0
#         for i in range(num_blocks):
#         # 提取每个块
#             Q_block = Q[i * block_size:(i + 1) * block_size, ...]
#             K_block = K[i * block_size:(i + 1) * block_size, ...]
#             # 计算当前块的注意力分数
#             attention_scores_block = torch.softmax(Q_block @ K_block.transpose(-2, -1), dim=-1)
#             attention_scores.append(attention_scores_block)
#             # print(attention_scores_block.shape)
#             left_ = i + 1
#         # 剩余的部分也要参与计算，不能舍弃
#         Q_block = Q[left_ * block_size:, ...]
#         K_block = K[left_ * block_size:, ...]
#         # 计算当前块的注意力分数
#         attention_scores_block = torch.softmax(Q_block @ K_block.transpose(-2, -1), dim=-1)
#         # print(attention_scores_block.shape)
#         block_list, padding_size = reshape_tensor(attention_scores_block, block_size)
#         for block in block_list:
#             attention_scores.append(block)
#         attention_scores = torch.cat(attention_scores, dim=0)
#         attention_scores = attention_scores[:x.shape[0], ...]
#         # print(attention_scores.shape)
#         # 加权聚合局部特征，同样也是分块计算处理
#         local_context = []
#         # print(attention_scores.shape)
#         #print(V.shape)
#         for i in range(num_blocks):
#             A_block = attention_scores[i * block_size:(i + 1) * block_size, ...]
#             V_block = V[i * block_size:(i + 1) * block_size, ...]
#             local_context_block = torch.matmul(A_block, V_block)
#             local_context.append(local_context_block)
            
#         # 最后一部分也需要重新计算
#         A_block = attention_scores[left_ * block_size:, ...]
#         V_block = V[left_ * block_size:, ...]
#         if A_block.shape[0] != 0:
#             indices = torch.randint(0, A_block.shape[0],(A_block.shape[1], ))
#             sub_tensor = V_block[indices, :]
#         #print(attention_scores.shape)
#         #print(A_block.shape)
#         #print(sub_tensor.shape)
#             local_context_block = torch.matmul(A_block, sub_tensor)
#             local_context.append(local_context_block)
#         local_context = torch.cat(local_context, dim=0)
#         #local_context = local_context[:attention_scores.shape[0], ...]
#         #print(local_context.shape)
#         return local_context