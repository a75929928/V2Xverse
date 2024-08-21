# -*- coding: utf-8 -*-
# Implementation of Select2Col.
# Author: Qian Huang <huangq@zhejianglab.com>, Yuntao Liu <liuyt@zhejianglab.com>
# License: TDG-Attribution-NonCommercial-NoDistrib
import time

import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.HPHA_fuse import HPHA
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
import torch

class PointPillarSelect2Col(nn.Module):
    def __init__(self, args):
        super(PointPillarSelect2Col, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = HPHA(args['HPHA_fusion'])
        self.multi_scale = args['HPHA_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1) # input dim 256 -> 128 
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

        self.shrink_conv_2 = DownsampleConv({'dim': [128], 'input_dim': 512, 'kernal_size': [3], 'padding': [1], 'stride': [1]})
        self.shrink_conv_3 = DownsampleConv({'dim': [128], 'input_dim': 128, 'kernal_size': [3], 'padding': [1], 'stride': [2]})

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features'] # torch.Size([104360, 32, 4])
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        time_delay = data_dict['time_delay'] ## add time delay for feature enhance
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176] # original size N = batch*num_cav
        spatial_features_2d = batch_dict['spatial_features_2d'] # torch.Size([N, 384, 96, 288])
        # Down-sample feature to reduce memory
        if self.shrink_flag: ## self.shrink_flag->True
            spatial_features_2d = self.shrink_conv(spatial_features_2d) # torch.Size([5, 256, 48, 144])
        psm_single = self.cls_head(spatial_features_2d) # psm_single torch.Size([5, 2, 48, 144])

        # Compressor, NOT USED NOW
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # add historical semantic information of ego
        # size of spatial_features [N, 64, 192, 576]
        # [batch_size, num_bev_features, grid_size y, grid_size x]
        if len(batch_dict['spatial_features'][3:].shape) >= 4:
            batch_semantic_informantion_dict = torch.cat(
                [batch_dict['spatial_features'][0].unsqueeze(0), batch_dict['spatial_features'][3:]], dim=0)
        else:
            batch_semantic_informantion_dict = torch.cat([batch_dict['spatial_features'][0].unsqueeze(0),
                                             batch_dict['spatial_features'][3:]].unsqueeze(0), dim=0)
        # Historical Prior Hybrid Attention based Fusion Net
        fused_feature = self.fusion_net(batch_semantic_informantion_dict, ## semantic information torch.Size([3, 64, 192, 576])
                                        batch_dict['spatial_features'][1:3], ## historical semantic information of ego
                                        psm_single, # [5, 2, 96, 288]
                                        record_len,
                                        pairwise_t_matrix,
                                        time_delay,
                                        self.backbone)
        if self.shrink_flag:
            # fused_feature (B, 384+128, 64, 144)
            fused_feature = self.shrink_conv_2(fused_feature) # [B, 512, 96, 288] -> [B, 128, 96, 288]
            # fused_feature = self.shrink_conv(fused_feature) # [B, 128, 96, 288]

        output_dict = {'fused_feature': fused_feature} # should fit with (batch_num, out_channel, 96, 288)

        fused_feature = self.shrink_conv_3(fused_feature) # [B, 128, 48, 144]

        cls_res = self.cls_head(fused_feature) # torch.Size([4, 2, 44, 144]) should match 24*144
        reg_res = self.reg_head(fused_feature) # torch.Size([4, 14, 44, 144])

        output_dict.update({'psm': cls_res, 'rm': reg_res})
        return output_dict
