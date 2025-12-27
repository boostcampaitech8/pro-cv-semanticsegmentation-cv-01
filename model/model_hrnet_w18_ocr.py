import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

# ==========================================
# OCR (Object-Contextual Representations) Modules
# ==========================================

class SpatialGather_Module(nn.Module):
    def __init__(self, cls_rule='softmax'):
        super(SpatialGather_Module, self).__init__()
        self.cls_rule = cls_rule

    def forward(self, feats, probs):
        batch_size, c, h, w = feats.size()
        batch_size, k, h, w = probs.size()
        input_x = feats.view(batch_size, c, -1).permute(0, 2, 1) # (B, HW, C)
        if self.cls_rule == 'softmax':
            probs = F.softmax(probs.view(batch_size, k, -1), dim=2)
        else:
            probs = F.sigmoid(probs.view(batch_size, k, -1))
        ocr_context = torch.matmul(probs, input_x).permute(0, 2, 1).unsqueeze(3) # (B, C, K, 1)
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else nn.Identity()
        
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels), nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1: x_pooled = self.pool(x)
        else: x_pooled = x
            
        query = self.f_pixel(x_pooled).view(batch_size, self.key_channels, -1).permute(0, 2, 1) # (B, HW, KeyC)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1) # (B, KeyC, K)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1).permute(0, 2, 1) # (B, K, KeyC)

        sim_map = torch.matmul(query, key) * (self.key_channels**-.5)
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x_pooled.size()[2:])
        context = self.f_up(context)
        
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return context

class OCRHead(nn.Module):
    def __init__(self, in_channels, num_classes, ocr_mid_channels=256, ocr_key_channels=128):
        # Reduced mid/key channels for W18 efficiency
        super(OCRHead, self).__init__()
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, ocr_mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ocr_mid_channels), nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module('softmax')
        self.ocr_distri_head = ObjectAttentionBlock(ocr_mid_channels, ocr_key_channels)
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        out_aux = self.aux_head(feats)
        feats_ocr = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats_ocr, out_aux)
        ocr_context = self.ocr_distri_head(feats_ocr, context)
        out = self.cls_head(feats_ocr + ocr_context)
        return out, out_aux

class HRNetOCR(nn.Module):
    def __init__(self, num_classes, backbone='hrnet_w18', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        self.in_channels = self.backbone.feature_info.channels()
        self.total_channels = sum(self.in_channels)
        
        # Consistent OCR Head
        self.ocr_head = OCRHead(
            in_channels=self.total_channels, 
            num_classes=num_classes,
            ocr_mid_channels=256, # Lighter for W18
            ocr_key_channels=128
        )

    def forward(self, x):
        feats = self.backbone(x)
        h0, w0 = feats[0].shape[2], feats[0].shape[3]
        upsampled = []
        for f in feats:
            if f.shape[2]!=h0 or f.shape[3]!=w0:
                f = F.interpolate(f, size=(h0, w0), mode='bilinear', align_corners=True)
            upsampled.append(f)
        feats_concat = torch.cat(upsampled, dim=1)
        out, out_aux = self.ocr_head(feats_concat)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

def get_model():
    return HRNetOCR(len(Config.CLASSES), backbone='hrnet_w18')
