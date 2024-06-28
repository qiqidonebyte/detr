# ResNet Backbone
# +------------+      +------------------------+
# |   Conv1    | ----> |   MaxPool (Downsample) |
# +------------+      +------------------------+
# |   Conv2    |      |   Residual Blocks      |
# +------------+      +------------------------+
# |   ...      |      |   ...                 |
# +------------+      +------------------------+
# |   ConvN    |      |   MaxPool (Downsample) |
# +------------+      +------------------------+
#
# (Intermediate Layer Getter)
# +--+----+----+----+----+---+
# | Layer 1 | Layer 2 | ... | Layer N |
# +--+----+----+----+----+---+
# |      |         |
# V      V         V
# +---------------------+     +---------------------+
# |   FPN Layer 1 (Top)  |     |   FPN Layer N (Bottom) |
# | 1x1 Conv, 256 Ch.    |     | 1x1 Conv, 256 Ch.    |
# +---------------------+     +---------------------+
# |      |               |      |
# V      V               V      V
# +-----------+     +-----------+     +-----------+
# | 1x1 Conv  | ... | 1x1 Conv  | ... | 1x1 Conv  |
# | 256 Ch.   |     | 256 Ch.   |     | 256 Ch.   |
# +-----------+     +-----------+     +-----------+
# |      |               |      |
# +------> Upsample <------+
# |      |
# V      V
# +--------+  +--------+
# |  P5    |  |  P2    |
# | (Top)  |  | (Low)  |
# +--------+  +--------+
#

from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork

from backbone import FrozenBatchNorm2d
from util.misc import NestedTensor, is_main_process


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
        self.backbone_channels = [self.backbone.layer2[-1].conv3.out_channels,
                                  self.backbone.layer3[-1].conv3.out_channels,
                                  self.backbone.layer4[-1].conv3.out_channels]
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.backbone_channels,
                                         out_channels=num_channels)
        # 定义卷积层，用于调整FPN输出的通道数，以匹配旧模型的输入要求
        self.channel_adapters = nn.ModuleDict({
            '0': nn.Conv2d(num_channels, num_channels, kernel_size=1),
            '1': nn.Conv2d(num_channels, num_channels, kernel_size=1),
            '2': nn.Conv2d(num_channels, num_channels, kernel_size=1)
        })

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        # 将特征通过FPN，得到多尺度特征图
        fpn_features = self.fpn(xs)
        print(f"fpn_features: {type(fpn_features)}")
        out: Dict[str, NestedTensor] = {}
        for name, x in fpn_features.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
