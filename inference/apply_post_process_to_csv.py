import sys
import os
import pandas as pd
import numpy as np
import cv2
import cv2
import json
from tqdm.auto import tqdm
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from utils import encode_mask_to_rle, decode_rle_to_mask
try:
    import inference.post_process as post_process
except ImportError:
    import post_process as post_process

def process_csv(csv_path, save_path, image_root):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    classes = Config.CLASSES
    
    # Group by image
    image_names = df['image_name'].unique()
    
    results = [] # List of dicts
    
    print(f"Processing {len(image_names)} images...")
    
    # Scan image root for all files to map filename -> full path
    print(f"Scanning images in {image_root}...")
    image_map = {}
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_map[f] = os.path.join(root, f)
    
    print(f"Found {len(image_map)} images.")
    
    for img_name in tqdm(image_names):
        # 1. Load Image
        if img_name in image_map:
            img_path = image_map[img_name]
        else:
            print(f"Warning: Image not found {img_name}")
            continue
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        # 2. Reconstruct Mask (C, H, W)
        img_df = df[df['image_name'] == img_name]
        
        # Initialize full mask (binary)
        # We will convert 0/1 to probability-like for CRF (0.1 / 0.9)
        masks = np.zeros((len(classes), H, W), dtype=np.float32)
        
        for idx, row in img_df.iterrows():
            cls_name = row['class']
            rle = row['rle']
            if isinstance(rle, str) and len(rle) > 0:
                mask = decode_rle_to_mask(rle, height=H, width=W)
                cls_idx = Config.CLASS2IND[cls_name]
                masks[cls_idx] = mask # 0 or 1
                
        # 3. Apply Post Processing
        
        # Morphology (Closing) - Cheap cleanup
        # Doing this BEFORE CRF might help CRF start with better shape
        # Or AFTER? Usually Morph is good for holes. CRF works on edges.
        # Let's do Morph first to fill holes.
        for c in range(len(classes)):
            if masks[c].max() > 0:
                 masks[c] = post_process.apply_morphology(masks[c])
        
        # DenseCRF
        # Convert binary to soft prob for CRF freedom
        # 0 -> 0.1, 1 -> 0.9 (Allow CRF to change decisions)
        probs = masks.copy()
        probs[probs == 0] = 0.05
        probs[probs == 1] = 0.95
        
        # Resize for speed? CRF on 2048 is slow. 
        # But we want high res boundary.
        # Let's try full res first.
        refined_probs = post_process.apply_dense_crf_multilabel(probs, image)
        
        # Threshold back to binary
        refined_masks = (refined_probs > 0.5).astype(np.uint8)
        
        # 4. Save results
        for c, cls_name in enumerate(classes):
            mask = refined_masks[c]
            if mask.sum() > 0:
                rle = encode_mask_to_rle(mask)
            else:
                rle = ""
                
            results.append({
                "image_name": img_name,
                "class": cls_name,
                "rle": rle
            })
            
    # Save
    new_df = pd.DataFrame(results)
    new_df.to_csv(save_path, index=False)
    print(f"Saved processed CSV to {save_path}")

if __name__ == "__main__":
    # =========================================================
    # [사용자 설정] 아래 경로를 직접 수정하세요
    # =========================================================
    INPUT_CSV = "outputs/WJH_054_ensemble3.csv"   # 처리할 CSV 파일 경로
    OUTPUT_CSV = "outputs/WJH_057_ensemble3_post.csv" # 저장할 CSV 파일 경로
    IMAGE_ROOT = Config.TEST_IMAGE_ROOT          # 테스트 이미지 폴더 (보통 수정 불필요)
    # =========================================================

    print(f">> Input: {INPUT_CSV}")
    print(f">> Output: {OUTPUT_CSV}")
    
    process_csv(INPUT_CSV, OUTPUT_CSV, IMAGE_ROOT)
