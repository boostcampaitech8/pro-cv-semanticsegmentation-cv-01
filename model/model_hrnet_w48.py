import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

class HRNetSegmentation(nn.Module):
    def __init__(self, num_classes, backbone='hrnet_w48', pretrained=True):
        super().__init__()
        # Load HRNet backbone from timm
        self.backbone = timm.create_model(
            backbone, 
            features_only=True, 
            pretrained=pretrained
        )
        
        # Determine total channels by inspecting the model
        # W48: [48, 96, 192, 384] -> Total 720
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
        # 1. Backbone Features
        feats = self.backbone(x)
        
        # 2. Upsample and Concatenate (HRNetV2 Strategy)
        h_target, w_target = feats[0].shape[2], feats[0].shape[3]
        
        upsampled_feats = []
        for feat in feats:
            if feat.shape[2] != h_target or feat.shape[3] != w_target:
                feat = F.interpolate(feat, size=(h_target, w_target), mode='bilinear', align_corners=True)
            upsampled_feats.append(feat)
        
        # Concat along channel axis
        out = torch.cat(upsampled_feats, dim=1) # (B, sum(all_channels), H_target, W_target)
        
        # 3. Final Head
        out = self.last_layer(out) # (B, num_classes, H_target, W_target)
        
        # 4. Upsample to original input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True) # (B, num_classes, H, W)
        
        return out

def get_model():
    model = HRNetSegmentation(
        num_classes=len(Config.CLASSES),
        backbone='hrnet_w48', # Largest Backbone
        pretrained=True
    )
    return model
