import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from config import Config
import json

# =============================================================================
# 1. 학습 환경 설정 관련 유틸리티 (Seed 설정 등)
# =============================================================================

def set_seed(seed):
    """
    랜덤 시드를 고정하여 실험의 재현성을 보장합니다.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# =============================================================================
# 2. 모델 저장 및 기본 메트릭 유틸리티
# =============================================================================

def save_model(model, saved_dir, file_name="best_model.pt"):
    """
    학습된 모델을 저장합니다. Multi-GPU 환경을 고려하여 래핑을 해제한 후 저장합니다.
    """
    # [Multi-GPU] 저장 전 DataParallel/DDP 래핑 해제
    model_to_save = model.module if hasattr(model, 'module') else model
    
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model_to_save, output_path)

def dice_coef(y_true, y_pred):
    """
    Dice Coefficient를 계산하여 모델의 성능을 평가합니다.
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# =============================================================================
# 3. RLE (Run-Length Encoding) 관련 유틸리티
# =============================================================================

def encode_mask_to_rle(mask):
    """
    이진 마스크를 RLE(Run-Length Encoding) 형식의 문자열로 변환합니다.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    """
    RLE 문자열을 다시 이진 마스크 형태의 numpy 배열로 변환합니다.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((height, width))
# =============================================================================
# 4. 손실 함수 (Loss Functions) 구현
# =============================================================================
# ※ 참고: 표준 BCE와 DICE Loss는 train.py에서 라이브러리 형태로 직접 구현됨

# 4.1 Focal Loss
class FocalLoss(nn.Module):
    """
    불균형한 클래스 분포를 다루기 위해 잘 맞추지 못하는 샘플에 높은 가중치를 부여하는 Loss입니다.
    """
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

# 4.2 Jaccard Loss (IoU Loss)
class JaccardLoss(nn.Module):
    """
    IoU (Intersection over Union) 기반의 Loss로, 영역 간의 겹침 정도를 최적화합니다.
    """
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

# 4.3 Tversky Loss
class TverskyLoss(nn.Module):
    """
    Dice와 Jaccard의 일반화된 버전으로, FP와 FN의 가중치를 더 조절할 수 있습니다.
    """
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

# 4.4 Generalized Dice Loss (GDL)
class GeneralizedDiceLoss(nn.Module):
    """
    다중 클래스 세분화에서 각 클래스의 볼륨에 따라 가중치를 조절하는 상호보완적 Dice Loss입니다.
    """
    def __init__(self, smooth=Config.GDL_SMOOTH, gamma=Config.GDL_GAMMA):
        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).flatten(2)
        targets = targets.flatten(2)
        target_sum = targets.sum(-1)
        # 픽셀 수의 gamma 승에 반비례하도록 클래스별 가중치 조절
        class_weights = 1.0 / (torch.pow(target_sum, self.gamma) + self.smooth)
        intersection = (inputs * targets).sum(-1)
        union = (inputs + targets).sum(-1)
        weighted_inter = (class_weights * intersection).sum(1)
        weighted_union = (class_weights * union).sum(1)
        gdl = (2. * weighted_inter + self.smooth) / (weighted_union + self.smooth)
        return 1 - gdl.mean()

# 4.5 Pixel Weighted BCE
class PixelWeightedBCE(nn.Module):
    """
    픽셀별 중요도(희소성 등)에 따라 BCE에 가중치를 부여하는 Loss입니다.
    """
    def __init__(self, smooth=Config.PW_BCE_SMOOTH):
        super(PixelWeightedBCE, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets_f = targets.flatten(2)
        pixel_counts = targets_f.sum(-1, keepdim=True).unsqueeze(-1)
        total_pixels = targets_f.shape[-1]
        weights = total_pixels / (pixel_counts + self.smooth)
        weights = torch.log(weights + 1.0) # 로그 스케일로 가중치 완만하게 적용
        weighted_loss = bce_loss * weights
        return weighted_loss.mean()

# 4.6 Overlap Penalty Loss
class OverlapPenaltyLoss(nn.Module):
    """
    해부학적으로 불가능한 장기 간의 과도한 중첩을 방지하기 위해 생성된 페널티 Loss입니다.
    """
    def __init__(self):
        super(OverlapPenaltyLoss, self).__init__()
        self.min_confidence = Config.OVERLAP_MIN_CONFIDENCE
        self.overlap_pairs = self._load_from_json()
        
    def _load_from_json(self):
        """JSON 파일에서 주요 중첩 쌍(평균 900px 이상) 정보를 로드합니다."""
        json_path = Config.OVERLAP_ANALYSIS_FILE
        
        # 파일 존재 확인
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ Overlap 분석 파일을 찾을 수 없습니다: {json_path}")
        
        # JSON 읽기
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 중첩 데이터 추출
        pairs = []
        for pair_info in data['pairs']:
            avg_pixels = pair_info['avg_overlap_pixels']
            
            # 평균 900픽셀 이상인 유의미한 중첩 페어만 선택
            if avg_pixels >= 900:
                cls_a, cls_b = pair_info['classes']
                pairs.append((cls_a, cls_b))
        
        if len(pairs) == 0:
            raise ValueError("❌ 평균 중첩 900픽셀 이상의 유효한 페어를 찾을 수 없습니다!")
        
        print(f">> [OverlapPenalty] Filtered Pairs (avg >= 900px) 로드 완료")
        return pairs
    
    def forward(self, inputs, targets):
        """
        inputs: [B, 29, H, W] - 로짓값
        targets: [B, 29, H, W] - 정답 라벨
        """
        inputs_sigmoid = torch.sigmoid(inputs)
        total_penalty = 0.0
        num_pairs_processed = 0
        
        for cls_a, cls_b in self.overlap_pairs:
            # 타겟 맵에서 실제로 중첩된 픽셀을 찾음
            overlap_mask = (targets[:, cls_a] * targets[:, cls_b]) > 0.5
            
            # 중첩된 픽셀이 10개 이상인 경우에만 계산 진행
            if overlap_mask.sum() > 10:
                # 두 클래스의 예측값(확률) 추출
                pred_a = inputs_sigmoid[:, cls_a][overlap_mask]
                pred_b = inputs_sigmoid[:, cls_b][overlap_mask]
                
                # 둘 중 예측 확률이 더 낮은 값을 기준으로 페널티 부여
                min_conf = torch.min(pred_a, pred_b)
                
                # 특정 확신도(min_confidence)보다 낮으면 페널티 가산
                penalty = F.relu(self.min_confidence - min_conf).mean()
                total_penalty += penalty
                num_pairs_processed += 1
        
        # 처리된 쌍이 있다면 전체 평균 페널티 반환
        if num_pairs_processed > 0:
            return total_penalty / num_pairs_processed
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)


# 4.7 Combined Loss
class CombinedLoss(nn.Module):
    """
    여러 가지 손실 함수를 가중 합산하여 최종 손실값을 산출합니다.
    (예: Dice + Focal, BCE + IoU 등)
    """
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
        
        # weight_c가 Config에 정의되어 있지 않을 경우 비중을 자동 계산
        if len(Config.LOSS_WEIGHTS) > 2:
            self.weight_c = Config.LOSS_WEIGHTS[2]
        else:
            self.weight_c = 1 - self.weight_a - self.weight_b 
        
    def forward(self, inputs, targets):
        total_loss = self.weight_a * self.loss_a(inputs, targets) + \
                     self.weight_b * self.loss_b(inputs, targets)
        
        # 추가적인 세 번째 loss_c가 있는 경우 가중치를 적용하여 더함
        if self.loss_c is not None:
            total_loss += self.weight_c * self.loss_c(inputs, targets)
        
        return total_loss