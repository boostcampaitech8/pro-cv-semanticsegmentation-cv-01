import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import os
import importlib
import numpy as np
import sys
import cv2

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from utils import encode_mask_to_rle

def get_hand_direction_from_mask(mask_sigmoid):
    """
    PA View 기준: 엄지(index 0)가 새끼(index 15)보다 왼쪽에 있으면 오른손(Right).
    """
    f1_mask = mask_sigmoid[0].detach().cpu().numpy()
    f16_mask = mask_sigmoid[15].detach().cpu().numpy()
    
    def get_center_x(m):
        coords = np.where(m > 0.5)
        if len(coords[1]) == 0: return -1
        return np.mean(coords[1])

    f1_x = get_center_x(f1_mask)
    f16_x = get_center_x(f16_mask)
    
    if f1_x == -1 or f16_x == -1: return "Right"
    return "Right" if f1_x < f16_x else "Left"

def test():
    # 1. 모듈 및 모델 로드
    try:
        dataset_module = importlib.import_module("dataset.dataset_flip")
    except ModuleNotFoundError:
        print("Error loading dataset_flip.py. Make sure it exists in dataset/ folder.")
        return
        
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    dir_model_path = "checkpoints/Base_UNet/best_model.pt"
    main_model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")

    if not os.path.exists(dir_model_path):
        print(f"Direction model not found at {dir_model_path}.")
        return
    if not os.path.exists(main_model_path):
        print(f"Main model not found at {main_model_path}.")
        return

    print(f"Loading Stage 1 Model (Direction) from {dir_model_path}")
    dir_model = torch.load(dir_model_path, map_location='cuda', weights_only=False)
    dir_model.eval()

    print(f"Loading Stage 2 Model (Main) from {main_model_path}")
    main_model = torch.load(main_model_path, map_location='cuda', weights_only=False)
    main_model.eval()

    # 2. 데이터 로더
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    results_dict = {}
    
    # 3. 인퍼런스 시작
    print("Start Two-stage Inference (Direction Detection -> Prediction)...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            images = images.cuda()
            
            # --- Stage 1: 방향 판별 ---
            with torch.amp.autocast(device_type="cuda"):
                dir_outputs = dir_model(images)
                if isinstance(dir_outputs, dict): dir_outputs = dir_outputs['out']
            dir_outputs = torch.sigmoid(dir_outputs)
            
            for i in range(len(image_names)):
                direction = get_hand_direction_from_mask(dir_outputs[i])
                is_left = (direction == "Left")
                
                curr_img = images[i:i+1]
                if is_left:
                    curr_img = torch.flip(curr_img, dims=[3])
                
                # --- Stage 2: 메인 모델 예측 ---
                with torch.amp.autocast(device_type="cuda"):
                    outputs = main_model(curr_img)
                    if isinstance(outputs, dict): outputs = outputs['out']
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                output_np = outputs[0].detach().cpu().numpy()
                
                if is_left:
                    output_np = output_np[:, :, ::-1]
                
                output_mask = (output_np > 0.5).astype(np.uint8)
                image_name = os.path.basename(image_names[i])
                
                for c in range(output_mask.shape[0]):
                    rle = encode_mask_to_rle(output_mask[c])
                    class_name = Config.CLASSES[c]
                    results_dict[f"{class_name}_{image_name}"] = rle

    # 4. 결과 저장
    print("Finalizing Submission...")
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = [results_dict.get(f"{row['class']}_{row['image_name']}", "") for _, row in sample_df.iterrows()]
        sample_df['rle'] = final_rles
        final_df = sample_df
    else:
        final_df = pd.DataFrame([
            {"image_name": k.split('_', 1)[1], "class": k.split('_', 1)[0], "rle": v}
            for k, v in results_dict.items()
        ])

    save_path = f"submission_{Config.EXPERIMENT_NAME}_flip.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    test()
