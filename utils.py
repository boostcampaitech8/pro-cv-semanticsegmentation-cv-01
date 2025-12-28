import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from config import Config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_model(model, saved_dir, file_name="best_model.pt"):
    # [Multi-GPU] Unwrap DataParallel/DDP before saving
    model_to_save = model.module if hasattr(model, 'module') else model
    
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model_to_save, output_path)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 0. BCE와 DICE는 train.py에서 라이브러리로 구현

# 1. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean': return torch.mean(f_loss)
        elif self.reduction == 'sum': return torch.sum(f_loss)
        else: return f_loss

# 2. Jaccard Loss
class JaccardLoss(nn.Module):
    def __init__(self, smooth=Config.LOSS_SMOOTH):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

# 3. Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=Config.TVERSKY_ALPHA, beta=Config.TVERSKY_BETA, smooth=Config.LOSS_SMOOTH):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - Tversky

# 4. Generalized Dice Loss
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=Config.GDL_SMOOTH, gamma=Config.GDL_GAMMA):
        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).flatten(2)
        targets = targets.flatten(2)
        target_sum = targets.sum(-1)
        # 픽셀 수의 gamma 승에 반비례하도록 가중치 조절
        class_weights = 1.0 / (torch.pow(target_sum, self.gamma) + self.smooth)
        intersection = (inputs * targets).sum(-1)
        union = (inputs + targets).sum(-1)
        weighted_inter = (class_weights * intersection).sum(1)
        weighted_union = (class_weights * union).sum(1)
        gdl = (2. * weighted_inter + self.smooth) / (weighted_union + self.smooth)
        return 1 - gdl.mean()

# 5. Pixel Weighted BCE
class PixelWeightedBCE(nn.Module):
    def __init__(self, smooth=Config.PW_BCE_SMOOTH):
        super(PixelWeightedBCE, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets_f = targets.flatten(2)
        pixel_counts = targets_f.sum(-1, keepdim=True).unsqueeze(-1)
        total_pixels = targets_f.shape[-1]
        weights = total_pixels / (pixel_counts + self.smooth)
        weights = torch.log(weights + 1.0) # 로그 스케일 가중치
        weighted_loss = bce_loss * weights
        return weighted_loss.mean()

# 6. Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, loss_a, loss_b, weight_a=Config.LOSS_WEIGHTS[0], weight_b=Config.LOSS_WEIGHTS[1]):
        super(CombinedLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        
    def forward(self, inputs, targets):
        return self.weight_a * self.loss_a(inputs, targets) + self.weight_b * self.loss_b(inputs, targets)