# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding, PositionEmbeddingSine


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        print(f"Shape before processing: {tensor_list.tensors.shape}, Type: {tensor_list.tensors.dtype}")
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
            print(out[name])
        print(out.keys())
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # print(f"Joiner Shape before processing: {tensor_list.tensors.shape}, Type: {tensor_list.tensors.dtype}")
        xs = self[0](tensor_list)  # xs is now a dictionary of feature maps
        # print(f"xs: {xs.keys()}")
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.fpn
    if args.fpn:
        backbone = BackboneWithFPN2(args.backbone, train_backbone,
                                    return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class BackboneWithFPN2(nn.Module):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        # 定义特征金字塔网络（FPN）
        self.backbone_channels = [backbone.layer1[-1].conv3.out_channels,
                                  backbone.layer2[-1].conv3.out_channels,
                                  backbone.layer3[-1].conv3.out_channels,
                                  backbone.layer4[-1].conv3.out_channels]
        # print(self.backbone_channels)
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.backbone_channels,
                                         out_channels=num_channels)
        # 定义卷积层，用于调整FPN输出的通道数，以匹配旧模型的输入要求, 此处fpn已经做了转换,如果没有则可以自行卷积转换通道
        self.channel_adapters = nn.ModuleDict({
            '0': nn.Conv2d(self.backbone_channels[0], num_channels, kernel_size=1),
            '1': nn.Conv2d(self.backbone_channels[1], num_channels, kernel_size=1),
            '2': nn.Conv2d(self.backbone_channels[2], num_channels, kernel_size=1),
            '3': nn.Conv2d(self.backbone_channels[2], num_channels, kernel_size=1)
        })

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        # print(f"xs : {type(xs)}, xs:{xs.keys()}")
        # 将特征通过FPN，得到多尺度特征图
        fpn_features = self.fpn(xs)
        # print(f"fpn_features: {type(fpn_features)}, fpn_features:{fpn_features.keys()}")
        out: Dict[str, NestedTensor] = {}
        for name, x in fpn_features.items():
            # 对x进行统一通道
            # print(f"name:{name}, x:{x.shape}")
            # x = self.channel_adapters[name](x)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
