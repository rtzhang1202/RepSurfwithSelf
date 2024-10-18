"""
Author: Haoxi Ran
Date: 06/30/2022
Modify: Xu in 24/9
Modify: Zhang in 24/10
"""

import torch
import torch.nn as nn
from modules.repsurface_utils import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD

'''
    这里在原有的基础上存在几个方面的调整：
        首先是原始模型的时候，可以把所有的超参数都除以2；
        然后是引入了编码器解码器用于特征编码；
        其次是引入了空间自我注意力机制，今年pytorch出了一个新的包在torch.nn里面，即MultiHeadAttention。之前搭建的时候是手动实现的，重新实现的代码选择直接用这个库，但是可能会pytorch版本和原有环境不同需要升级一下pytorch版本。印象中升级了更好的版本是不影响的。

    可以通过注释掉某些模块的内容将对应的部分去掉。需要注意的是，如果只使用了自注意力的话，还需要将自适应加权的部分也注释掉。
    但如果说只使用编码器的话，自适应加权的部分需要把attention_output改为self.fp1的输出，剩下的不变。
        
        
'''
class MLPs(nn.Module):
    def __init__(self, prev_channel, skip_channel, mlp) -> None:
        super(MLPs, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        self.skip      = skip_channel is not None
        self.mlp_f0    = nn.Linear(prev_channel, mlp[0])
        self.norm_f0   = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_s0  = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel   = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        center_channel = 6 if args.return_polar else 3
        repsurf_in_channel = 10
        repsurf_out_channel = 10

        # SurfaceAbstractionCD类用于抽象和特征提取，逐步减少点云的数量，同时保留重要的几何和特征信息
        self.sa1 = SurfaceAbstractionCD(4, 32, args.in_channel + repsurf_out_channel, center_channel, [32, 32, 64],
                                        True, args.return_polar, num_sector=4)
        self.sa2 = SurfaceAbstractionCD(4, 32, 64 + repsurf_out_channel, center_channel, [64, 64, 128],
                                        True, args.return_polar)
        self.sa3 = SurfaceAbstractionCD(4, 32, 128 + repsurf_out_channel, center_channel, [128, 128, 256],
                                        True, args.return_polar)
        self.sa4 = SurfaceAbstractionCD(4, 32, 256 + repsurf_out_channel, center_channel, [256, 256, 512],
                                        True, args.return_polar)

        self.fp4 = SurfaceFeaturePropagationCD(512, 256, [256, 256])
        self.fp3 = SurfaceFeaturePropagationCD(256, 128, [256, 256])
        self.fp2 = SurfaceFeaturePropagationCD(256, 64, [256, 128])
        self.fp1 = SurfaceFeaturePropagationCD(128, None, [128, 128, 128])

        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5), # 这里在实验的时候把dropout调低一点模型的表现会有提升
            nn.Linear(128, args.num_class),
        )

        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_in_channel, repsurf_out_channel)

        # 编码器结构声明
        self.input_dim  = 128
        self.hidden_dim = 64
        self.latent_dim = 32
        self.output_dim = 128
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,  self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        # MLPs声明
        self.enfp1 = MLPs(32, None, [32, 32])
        self.enfp2 = MLPs(32, None, [32, 16])
        self.enfp3 = MLPs(16, None, [16, 32])
        self.enfp4 = MLPs(32, None, [32, 32, 32])


        # 这里直接使用pytorch新版本提出的MultiheadAttention库自动快速实现空间自我注意力机制，没有自己实现
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1, batch_first=True)

        self.Weight1   = nn.Linear(128,128)
        self.Weight2   = nn.Linear(128,128)

    def forward(self, pos_feat_off0):
        pos_nor_feat_off0 = [
            pos_feat_off0[0],
            self.surface_constructor(pos_feat_off0[0], pos_feat_off0[2]),
            torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1),
            pos_feat_off0[2]
        ]
        
        # 点云特征的抽象和简化
        pos_nor_feat_off1 = self.sa1(pos_nor_feat_off0)
        pos_nor_feat_off2 = self.sa2(pos_nor_feat_off1)
        pos_nor_feat_off3 = self.sa3(pos_nor_feat_off2)
        pos_nor_feat_off4 = self.sa4(pos_nor_feat_off3)

        del pos_nor_feat_off0[1], pos_nor_feat_off1[1], pos_nor_feat_off2[1], pos_nor_feat_off3[1], pos_nor_feat_off4[1]
        pos_nor_feat_off3[1] = self.fp4(pos_nor_feat_off3, pos_nor_feat_off4)
        pos_nor_feat_off2[1] = self.fp3(pos_nor_feat_off2, pos_nor_feat_off3)
        pos_nor_feat_off1[1] = self.fp2(pos_nor_feat_off1, pos_nor_feat_off2)
        pos_nor_feat_off0[1] = self.fp1([pos_nor_feat_off0[0], None, pos_nor_feat_off0[2]], pos_nor_feat_off1)

        # 编码器对输入特征编码，并使用MLPs深入提取特征
        encoding = self.encoder(pos_nor_feat_off0[1])
        encoding = self.enfp1(encoding)
        encoding = self.enfp2(encoding)
        encoding = self.enfp3(encoding)
        encoding = self.enfp4(encoding)
        decoding = self.decoder(encoding)

        # 空间自我注意力机制放在了最后，用于捕获全局的注意力信息
        # 这里的空间自我注意力模块还可以放在前面，用于预先调整特征的权重进而让模型更好的学习到终点的特征
        attention_output, _ = self.attention(pos_nor_feat_off0[1], pos_nor_feat_off0[1], pos_nor_feat_off0[1])

        # 自适应更新二者加权的权重
        weight1 = self.Weight1(attention_output)
        weight2 = self.Weight2(decoding)
        feature = self.classifier(attention_output*weight1 + decoding*weight2)

        return feature
    
