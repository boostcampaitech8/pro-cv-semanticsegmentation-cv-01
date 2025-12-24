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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from utils import encode_mask_to_rle

def test():
    # 1. 모듈 및 모델 로드
    try:
        dataset_module = importlib.import_module("dataset.dataset_crop")
    except ModuleNotFoundError:
        print("Error loading dataset_crop module.")
        return
        
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    # Stage 1 Model (Base UNet for BBox)
    stage1_path = "checkpoints/Base_UNet/best_model.pt"
    # Stage 2 Model (Current Best Model)
    stage2_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    
    if not os.path.exists(stage2_path):
        print(f"Stage 2 Model not found: {stage2_path}")
        return

    print(f"Loading Stage 1 Model from {stage1_path}")
    model1 = torch.load(stage1_path, map_location='cuda', weights_only=False)
    model1.eval()

    print(f"Loading Stage 2 Model from {stage2_path}")
    model2 = torch.load(stage2_path, map_location='cuda', weights_only=False)
    model2.eval()

    # 2. 데이터셋 (Staging)
    # 이미지 원본을 들고와서 Stage 1에 넣어야 하므로 트랜스폼 정의 확인
    tf = get_transforms(is_train=False)
    
    test_image_root = Config.TEST_IMAGE_ROOT
    filenames = np.array(sorted([
        os.path.relpath(os.path.join(root, fname), start=test_image_root)
        for root, _dirs, files in os.walk(test_image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    ]))
    
    results_dict = {}
    
    # 3. 인퍼런스 시작
    print("Start 2-Stage Inference...")
    with torch.no_grad():
        for fname in tqdm(filenames):
            image_path = os.path.join(test_image_root, fname)
            image = cv2.imread(image_path)
            orig_h, orig_w = image.shape[:2]
            
            # --- [Stage 1] Get BBox ---
            # 원본 -> 512x512 -> Stage 1 모델 -> Mask -> BBox
            input_st1 = tf(image=image)["image"]
            input_st1 = input_st1.transpose(2, 0, 1)
            input_st1 = torch.from_numpy(input_st1).float().unsqueeze(0).cuda()
            with torch.amp.autocast(device_type="cuda"):
                out1 = model1(input_st1)
                if isinstance(out1, dict): out1 = out1['out']
                out1 = (torch.sigmoid(out1) > 0.5).float().sum(dim=1) # Combine all channels
                out1 = (out1 > 0).cpu().numpy()[0].astype(np.uint8)
            
            # Resize mask back to find accurate bbox in original scale
            mask1 = cv2.resize(out1, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                all_cnt = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_cnt)
                
                # Make it square with padding
                padding = 50
                side = max(w, h) + padding * 2
                cx, cy = x + w/2, y + h/2
                
                x_min = int(cx - side/2)
                y_min = int(cy - side/2)
                x_max = int(cx + side/2)
                y_max = int(cy + side/2)
            else:
                # Fallback if Stage 1 fails to find anything
                x_min, y_min, x_max, y_max = 0, 0, orig_w, orig_h
                
            # --- [Stage 2] Segment Cropped ---
            cropped_image = dataset_module.crop_with_padding(image, x_min, y_min, x_max, y_max)
            cropped_h, cropped_w = cropped_image.shape[:2]
            
            input_st2 = tf(image=cropped_image)["image"]
            input_st2 = input_st2.transpose(2, 0, 1)
            input_st2 = torch.from_numpy(input_st2).float().unsqueeze(0).cuda()
            with torch.amp.autocast(device_type="cuda"):
                out2 = model2(input_st2)
                if isinstance(out2, dict): out2 = out2['out']
                out2 = torch.sigmoid(out2) # (1, 29, 512, 512)
            
            # Restore to original scale
            out2 = F.interpolate(out2, size=(cropped_h, cropped_w), mode="bilinear")
            out2 = (out2.squeeze(0) > 0.5).detach().cpu().numpy() # (29, cH, cW)
            
            image_name = os.path.basename(fname)
            for c in range(len(Config.CLASSES)):
                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                
                # 역으로 복원: crop_with_padding 로직의 반대
                # Valid range in original image
                src_x1, src_y1 = max(0, x_min), max(0, y_min)
                src_x2, src_y2 = min(orig_w, x_max), min(orig_h, y_max)
                
                # Range in cropped mask
                dst_x1, dst_y1 = max(0, -x_min), max(0, -y_min)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                if src_x1 < src_x2 and src_y1 < src_y2:
                    full_mask[src_y1:src_y2, src_x1:src_x2] = out2[c][dst_y1:dst_y2, dst_x1:dst_x2]
                
                if full_mask.shape != (2048, 2048):
                    full_mask = cv2.resize(full_mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)
                
                rle = encode_mask_to_rle(full_mask)
                results_dict[f"{Config.CLASSES[c]}_{image_name}"] = rle

    # 4. Matching with sample_submission.csv
    print("Post-processing and Matching with sample_submission.csv...")
    sample_sub_path = "sample_submission.csv"
    
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        final_df = sample_df
    else:
        print("Warning: sample_submission.csv not found!")
        final_df = pd.DataFrame([
            {"image_name": k.split('_', 1)[1], "class": k.split('_', 1)[0], "rle": v}
            for k, v in results_dict.items()
        ])

    # 5. 최종 저장
    save_path = f"submission_{Config.EXPERIMENT_NAME}_crop.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    test()
