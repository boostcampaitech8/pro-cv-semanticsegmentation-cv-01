import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import importlib
import numpy as np
import cv2
from tqdm.auto import tqdm

from config import Config

# 시각화할 샘플 개수
NUM_SAMPLES = 5 
# 저장할 폴더
SAVE_DIR = "visualize_results"

def decode_mask_to_colormap(mask):
    """
    (29, H, W) 형태의 마스크를 알록달록한 (H, W, 3) 이미지로 변환
    """
    H, W = mask.shape[1], mask.shape[2]
    colormap = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 클래스별 고유 색상 생성 (랜덤하지만 고정된 시드)
    np.random.seed(42)
    colors = [np.random.randint(0, 255, 3).tolist() for _ in range(len(Config.CLASSES))]
    
    # 마스크가 있는 부분을 색칠
    for c in range(len(Config.CLASSES)):
        class_mask = mask[c] # (H, W)
        if class_mask.max() > 0:
            color = colors[c]
            # 해당 클래스 영역에 색상 입히기 (살짝 섞이도록)
            colormap[class_mask > 0.5] = color
            
    return colormap

def visualize():
    # 1. 설정 및 모델 로드
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
        XRayDataset = dataset_module.XRayDataset
        get_transforms = dataset_module.get_transforms
    except ModuleNotFoundError:
        print("Error loading modules. Check config.py")
        return

    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("학습이 완료된 best_model.pt 파일이 필요합니다.")
        return

    print(f">> Loading Model from {model_path}")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()

    # 2. Validation 데이터셋 로드 (정답 비교용)
    valid_dataset = XRayDataset(is_train=False, transforms=get_transforms(is_train=False))
    
    # 랜덤 샘플링을 위해 Shuffle=True
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2)

    print(f">> Start Visualization for {NUM_SAMPLES} samples...")
    
    count = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(valid_loader):
            if count >= NUM_SAMPLES:
                break
                
            images = images.cuda()
            masks = masks.cuda() # (B, 29, H, W)

            # 추론
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
            
            # 후처리
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float() # Threshold 0.5
            
            # --- 시각화 준비 ---
            # 1. 원본 이미지 (C, H, W) -> (H, W, C)
            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            # 정규화 해제 (시각화를 위해 대략적인 복원)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8)
            # 만약 흑백 1채널이면 RGB로 변환
            if img_np.shape[2] == 1:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # OpenCV 호환

            # 2. 정답 마스크 (Ground Truth) -> 컬러맵
            mask_np = masks[0].cpu().numpy()
            gt_colormap = decode_mask_to_colormap(mask_np)
            
            # 3. 예측 마스크 (Prediction) -> 컬러맵
            pred_np = preds[0].cpu().numpy()
            pred_colormap = decode_mask_to_colormap(pred_np)
            
            # 4. 오버레이 (원본 + 마스크)
            alpha = 0.5
            gt_overlay = cv2.addWeighted(img_np, 1, gt_colormap, alpha, 0)
            pred_overlay = cv2.addWeighted(img_np, 1, pred_colormap, alpha, 0)

            # --- Plotting ---
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(gt_overlay)
            plt.title("Ground Truth (Answer)")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_overlay)
            plt.title("Model Prediction")
            plt.axis("off")
            
            save_path = os.path.join(SAVE_DIR, f"sample_{count}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved visualization: {save_path}")
            count += 1

    print("Done! Check 'visualize_results' folder.")

if __name__ == "__main__":
    visualize()