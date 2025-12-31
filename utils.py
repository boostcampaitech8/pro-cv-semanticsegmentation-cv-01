import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from config import Config
import json

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

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((height, width))
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

# 6. Overlap Penalty Loss
class OverlapPenaltyLoss(nn.Module):
    def __init__(self):
        super(OverlapPenaltyLoss, self).__init__()
        self.min_confidence = Config.OVERLAP_MIN_CONFIDENCE
        self.overlap_pairs = self._load_from_json()
        
    def _load_from_json(self):
        """JSON 파일에서 overlap pairs 로드 (평균 900px 이상만)"""
        json_path = Config.OVERLAP_ANALYSIS_FILE
        
        # 파일 존재 확인
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ Overlap analysis file not found: {json_path}")
        
        # JSON 읽기
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # pairs 추출 (900px 이상만)
        pairs = []
        for pair_info in data['pairs']:
            avg_pixels = pair_info['avg_overlap_pixels']
            
            # 평균 900픽셀 이상만 선택
            if avg_pixels >= 900:
                cls_a, cls_b = pair_info['classes']
                pairs.append((cls_a, cls_b))
        
        if len(pairs) == 0:
            raise ValueError("❌ No overlap pairs found with avg >= 900 pixels!")
        
        print(f"  (Filtered: avg_overlap_pixels >= 900)")
        return pairs
    
    def forward(self, inputs, targets):
        """
        inputs: [B, 29, H, W] - logits
        targets: [B, 29, H, W] - 0 or 1
        """
        inputs_sigmoid = torch.sigmoid(inputs)
        total_penalty = 0.0
        num_pairs_processed = 0
        
        for cls_a, cls_b in self.overlap_pairs:
            # 겹치는 픽셀 마스크 생성
            overlap_mask = (targets[:, cls_a] * targets[:, cls_b]) > 0.5
            
            # 겹치는 픽셀이 충분히 있는지 확인
            if overlap_mask.sum() > 10:
                # 두 클래스의 예측값 추출
                pred_a = inputs_sigmoid[:, cls_a][overlap_mask]
                pred_b = inputs_sigmoid[:, cls_b][overlap_mask]
                
                # 둘 중 더 낮은 값 (약한 쪽)
                min_conf = torch.min(pred_a, pred_b)
                
                # min_confidence보다 낮으면 페널티
                penalty = F.relu(self.min_confidence - min_conf).mean()
                total_penalty += penalty
                num_pairs_processed += 1
        
        # 평균 페널티 반환 (처리된 쌍이 있는 경우)
        if num_pairs_processed > 0:
            return total_penalty / num_pairs_processed
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)


# 7. Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, loss_a, loss_b, loss_c=None, 
                 weight_a=Config.LOSS_WEIGHTS[0], 
                 weight_b=Config.LOSS_WEIGHTS[1], 
                 weight_c=None):
        super(CombinedLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.loss_c = loss_c
        self.weight_a = weight_a
        self.weight_b = weight_b
        
        # weight_c 처리
        if len(Config.LOSS_WEIGHTS) > 2:
            self.weight_c = Config.LOSS_WEIGHTS[2]
        else:
            self.weight_c = 1 - self.weight_a - self.weight_b 
        
    def forward(self, inputs, targets):
        total_loss = self.weight_a * self.loss_a(inputs, targets) + \
                     self.weight_b * self.loss_b(inputs, targets)
        
        # loss_c가 있으면 추가
        if self.loss_c is not None:
            total_loss += self.weight_c * self.loss_c(inputs, targets)
        
        return total_loss