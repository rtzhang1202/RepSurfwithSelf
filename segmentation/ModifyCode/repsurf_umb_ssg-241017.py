"""
Author: Haoxi Ran
Date: 06/30/2022
Modify: Xu in 24/9
Modify: Zhang in 24/10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.repsurface_utils import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD

# 自定义的自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 嵌入的维度
        self.heads = heads  # 注意力头的数量
        self.head_dim = embed_size // heads  # 每个注意力头的维度

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        # 定义值、键、查询的线性变换
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # 输出的全连接层
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]  # 批次大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将嵌入分割为多个注意力头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 应用线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 注意力机制的计算
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # 计算能量值
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # 应用softmax获取注意力权重
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)  # 应用最后的线性变换
        return out

# 多层感知机类
class MLPs(nn.Module):
    def __init__(self, prev_channel, skip_channel, mlp):
        super(MLPs, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None  # 是否有跳跃连接
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

# 主模型类
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        center_channel = 6 if args.return_polar else 3
        repsurf_in_channel = 10
        repsurf_out_channel = 10

        from modules.repsurface_utils import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD

        # 点云特征抽象和传播模块，SurfaceAbstractionCD类用于抽象和特征提取，逐步减少点云的数量，同时保留重要的几何和特征信息
        self.sa1 = SurfaceAbstractionCD(4, 32, args.in_channel + repsurf_out_channel, center_channel, [32, 32, 64], True, args.return_polar, num_sector=4)
        self.sa2 = SurfaceAbstractionCD(4, 32, 64 + repsurf_out_channel, center_channel, [64, 64, 128], True, args.return_polar)
        self.sa3 = SurfaceAbstractionCD(4, 32, 128 + repsurf_out_channel, center_channel, [128, 128, 256], True, args.return_polar)
        self.sa4 = SurfaceAbstractionCD(4, 32, 256 + repsurf_out_channel, center_channel, [256, 256, 512], True, args.return_polar)

        # 特征传播模块
        self.fp4 = SurfaceFeaturePropagationCD(512, 256, [256, 256])
        self.fp3 = SurfaceFeaturePropagationCD(256, 128, [256, 256])
        self.fp2 = SurfaceFeaturePropagationCD(256, 64, [256, 128])
        self.fp1 = SurfaceFeaturePropagationCD(128, None, [128, 128, 128])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, args.num_class),
        )

        # 表面构造器
        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_in_channel, repsurf_out_channel)

        # 编码器和解码器
        self.input_dim = 128
        self.hidden_dim = 64
        self.latent_dim = 32
        self.output_dim = 128
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # MLPs结构
        self.enfp1 = MLPs(32, None, [32, 32])
        self.enfp2 = MLPs(32, None, [32, 16])
        self.enfp3 = MLPs(16, None, [16, 32])
        self.enfp4 = MLPs(32, None, [32, 32, 32])

        # 自注意力模块
        self.attention = SelfAttention(embed_dim=128, heads=4)

        # 权重调整层
        self.Weight1 = nn.Linear(128, 128)
        self.Weight2 = nn.Linear(128, 128)

    def forward(self, pos_feat_off0):
        pos_nor_feat_off0 = [
            pos_feat_off0[0],
            self.surface_constructor(pos_feat_off0[0], pos_feat_off0[2]),
            torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1),
            pos_feat_off0[2]
        ]

        # 特征抽象和传播
        pos_nor_feat_off1 = self.sa1(pos_nor_feat_off0)
        pos_nor_feat_off2 = self.sa2(pos_nor_feat_off1)
        pos_nor_feat_off3 = self.sa3(pos_nor_feat_off2)
        pos_nor_feat_off4 = self.sa4(pos_nor_feat_off3)

        # 删除不再需要的中间数据以节省内存
        del pos_nor_feat_off0[1], pos_nor_feat_off1[1], pos_nor_feat_off2[1], pos_nor_feat_off3[1], pos_nor_feat_off4[1]
        pos_nor_feat_off3[1] = self.fp4(pos_nor_feat_off3, pos_nor_feat_off4)
        pos_nor_feat_off2[1] = self.fp3(pos_nor_feat_off2, pos_nor_feat_off3)
        pos_nor_feat_off1[1] = self.fp2(pos_nor_feat_off1, pos_nor_feat_off2)
        pos_nor_feat_off0[1] = self.fp1([pos_nor_feat_off0[0], None, pos_nor_feat_off0[2]], pos_nor_feat_off1)

        # 编码、解码和特征提取
        encoding = self.encoder(pos_nor_feat_off0[1])
        encoding = self.enfp1(encoding)
        encoding = self.enfp2(encoding)
        encoding = self.enfp3(encoding)
        encoding = self.enfp4(encoding)
        decoding = self.decoder(encoding)

        # 应用自注意力模块
        attention_output = self.attention(pos_nor_feat_off0[1], pos_nor_feat_off0[1], pos_nor_feat_off0[1])

        # 权重调整和特征融合
        weight1 = self.Weight1(attention_output)
        weight2 = self.Weight2(decoding)
        feature = self.classifier(attention_output * weight1 + decoding * weight2)

        return feature