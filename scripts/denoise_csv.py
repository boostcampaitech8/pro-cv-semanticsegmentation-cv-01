```python
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import cv2 # cv2 is still used later in the code
from tqdm import tqdm
from config import Config # Added as per instruction
from utils import decode_rle_to_mask, encode_mask_to_rle # Swapped order as per instruction
# Use existing utils

def post_process_csv(input_csv, output_csv, threshold=400):
    df = pd.read_csv(input_csv)
    new_rles = []

    for rle in tqdm(df['rle'], desc="Cleaning CSV"):
        if pd.isna(rle) or rle == "":
            new_rles.append("")
            continue
        
        # 1. RLE를 마스크(2048x2048)로 복원
        mask = decode_rle_to_mask(rle, height=2048, width=2048)
        
        # 2. 작은 덩어리 제거 (Connected Components Analysis)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        clean_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= threshold:
                clean_mask[labels == i] = 1
        
        # 3. 다시 RLE로 변환
        new_rles.append(encode_mask_to_rle(clean_mask))
    
    df['rle'] = new_rles
    df.to_csv(output_csv, index=False)
    print(f"Finished! Saved to {output_csv}")

# 실행
post_process_csv("submission_CSB_012_segb3_4del_fixinf.csv", 
                 "submission_CSB_013.csv", 
                 threshold=400)