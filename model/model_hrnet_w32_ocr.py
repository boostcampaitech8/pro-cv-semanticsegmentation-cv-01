import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config

# ==========================================
# OCR (Object-Contextual Representations) Modules
# ==========================================

class SpatialGather_Module(nn.Module):
    """
    Aggregate features based on soft object regions (soft labels).
    """
    def __init__(self, cls_rule='softmax'):
        super(SpatialGather_Module, self).__init__()
        self.cls_rule = cls_rule

    def forward(self, feats, probs):
        batch_size, c, h, w = feats.size()
        batch_size, k, h, w = probs.size()
        
        # Softmax to get soft region masks
        input_x = feats.view(batch_size, c, -1)
        input_x = input_x.permute(0, 2, 1) # (B, HW, C)
        
        if self.cls_rule == 'softmax':
            probs = F.softmax(probs.view(batch_size, k, -1), dim=2) # (B, K, HW)
        else:
            probs = F.sigmoid(probs.view(batch_size, k, -1))
            
        # Object Region Representations (Matrix Mult)
        # (B, K, HW) @ (B, HW, C) -> (B, K, C)
        ocr_context = torch.matmul(probs, input_x)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3) # (B, C, K, 1)
        
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    """
    Compute Object Context via Attention between Features and Object Representations.
    """
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        
        if self.scale > 1:
            x = self.pool(x)

        # 1. Query: Pixel Representations (Upsampled if pooled)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1) # (B, KeyC, HW)
        query = query.permute(0, 2, 1) # (B, HW, KeyC)
        
        # 2. Key: Object Representations
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1) # (B, KeyC, K)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1) # (B, KeyC, K)
        value = value.permute(0, 2, 1) # (B, K, KeyC)

        # 3. Attention
        sim_map = torch.matmul(query, key) # (B, HW, K)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # 4. Context
        context = torch.matmul(sim_map, value) # (B, HW, KeyC)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context) # (B, InC, H, W)
        
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)

        return context

class OCRHead(nn.Module):
    def __init__(self, in_channels, num_classes, ocr_mid_channels=512, ocr_key_channels=256):
        super(OCRHead, self).__init__()
        
        # Auxiliary Head (Soft Object Region Predictor)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # OCR Module
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(cls_rule='softmax')
        self.ocr_distri_head = ObjectAttentionBlock(ocr_mid_channels, ocr_key_channels)
        
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feats):
        # feats: (B, C, H, W) - High Res Features
        
        # 1. Auxiliary Loss Input / Soft Object Regions
        out_aux = self.aux_head(feats) # (B, K, H, W)
        
        # 2. OCR Context
        feats_ocr = self.conv3x3_ocr(feats) # (B, MidC, H, W)
        context = self.ocr_gather_head(feats_ocr, out_aux) # (B, MidC, K, 1)
        ocr_context = self.ocr_distri_head(feats_ocr, context) # (B, MidC, H, W)
        
        # 3. Final Representation (Feature + Context)
        out = self.cls_head(feats_ocr + ocr_context)
        
        return out, out_aux

class HRNetOCR(nn.Module):
    def __init__(self, num_classes, backbone='hrnet_w32', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        self.in_channels = self.backbone.feature_info.channels() # [32, 64, 128, 256] for w32
        
        # HRNetV2 Representation (Concat all scales)
        self.total_channels = sum(self.in_channels)
        
        # OCR Head
        # W32 -> Total 480 channels
        self.ocr_head = OCRHead(
            in_channels=self.total_channels, 
            num_classes=num_classes,
            ocr_mid_channels=512,
            ocr_key_channels=256
        )

    def forward(self, x):
        # 1. Backbone Features
        feats = self.backbone(x)
        
        # 2. Upsample and Concat (HRNetV2)
        h0, w0 = feats[0].shape[2], feats[0].shape[3]
        upsampled_feats = []
        for feat in feats:
            if feat.shape[2] != h0 or feat.shape[3] != w0:
                feat = F.interpolate(feat, size=(h0, w0), mode='bilinear', align_corners=True)
            upsampled_feats.append(feat)
        feats_concat = torch.cat(upsampled_feats, dim=1) # (B, TotalC, H/4, W/4)
        
        # 3. OCR Head
        out, out_aux = self.ocr_head(feats_concat)
        
        # 4. Upsample to original size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        # We can also return out_aux if we want to train with aux loss, 
        # but for simplicity in current train_loop (which expects single output), we return main output.
        # Ideally, we should add Aux Loss.
        
        return out

def get_model():
    model = HRNetOCR(
        num_classes=len(Config.CLASSES),
        backbone='hrnet_w32', # Lighter Backbone
        pretrained=True
    )
    return model
