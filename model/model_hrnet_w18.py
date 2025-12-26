import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

class HRNetSegmentation(nn.Module):
    def __init__(self, num_classes, backbone='hrnet_w18', pretrained=True):
        super().__init__()
        # Load HRNet backbone from timm
        # features_only=True returns features from 4 stages
        self.backbone = timm.create_model(
            backbone, 
            features_only=True, 
            pretrained=pretrained
        )
        
        # Determine total channels by inspecting the model
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            feats = self.backbone(dummy)
            total_channels = sum([f.shape[1] for f in feats])
        
        # Projection/Mixing layer
        self.last_layer = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        h_target, w_target = feats[0].shape[2], feats[0].shape[3]
        
        upsampled_feats = []
        for feat in feats:
            if feat.shape[2] != h_target or feat.shape[3] != w_target:
                feat = F.interpolate(feat, size=(h_target, w_target), mode='bilinear', align_corners=True)
            upsampled_feats.append(feat)
        
        out = torch.cat(upsampled_feats, dim=1)
        out = self.last_layer(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

def get_model():
    model = HRNetSegmentation(
        num_classes=len(Config.CLASSES),
        backbone='hrnet_w18',
        pretrained=True
    )
    return model
