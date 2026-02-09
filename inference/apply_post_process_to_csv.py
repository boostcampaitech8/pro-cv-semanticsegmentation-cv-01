import sys
import os
import pandas as pd
import numpy as np
import cv2
import json
from tqdm.auto import tqdm
import math
import multiprocessing

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from utils import encode_mask_to_rle, decode_rle_to_mask

try:
    import inference.post_process as post_process
except ImportError:
    import post_process as post_process

def process_one_image(img_name, img_path, rle_dict, classes):
    """
    Worker function to process a single image.
    rle_dict: {class_name: rle_string}
    """
    try:
        image = cv2.imread(img_path)
        if image is None:
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        # Initialize full mask (binary)
        masks = np.zeros((len(classes), H, W), dtype=np.float32)
        
        for cls_name, rle in rle_dict.items():
            if isinstance(rle, str) and len(rle) > 0:
                mask = decode_rle_to_mask(rle, height=H, width=W)
                cls_idx = Config.CLASS2IND[cls_name]
                masks[cls_idx] = mask # 0 or 1
        
        # Apply Post Processing
        
        # 1. Morphology
        # Disabled: Causing performance drop by merging joints
        # for c in range(len(classes)):
        #     if masks[c].max() > 0:
        #          masks[c] = post_process.apply_morphology(masks[c])
        
        # 2. DenseCRF
        # 0 -> 0.05, 1 -> 0.95
        probs = masks.copy()
        probs[probs == 0] = 0.05
        probs[probs == 1] = 0.95
        
        refined_probs = post_process.apply_dense_crf_multilabel(probs, image)
        
        # Threshold back to binary
        refined_masks = (refined_probs > 0.5).astype(np.uint8)
        
        # Encode results
        img_results = []
        for c, cls_name in enumerate(classes):
            mask = refined_masks[c]
            if mask.sum() > 0:
                rle = encode_mask_to_rle(mask)
            else:
                rle = ""
            
            # Only append if we want to keep all rows (even empty) - usually submission requires all
            img_results.append({
                "image_name": img_name,
                "class": cls_name,
                "rle": rle
            })
            
        return img_results
        
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return []

def process_one_image_wrapper(args):
    return process_one_image(*args)

def process_csv(csv_path, save_path, image_root):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    classes = Config.CLASSES
    
    # Pre-scan images
    print(f"Scanning images in {image_root}...")
    image_map = {}
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_map[f] = os.path.join(root, f)
    
    # Prepare arguments for multiprocessing
    image_names = df['image_name'].unique()
    tasks = []
    
    print(f"Preparing tasks for {len(image_names)} images...")
    
    for img_name in image_names:
        if img_name not in image_map:
            print(f"Warning: Image not found {img_name}")
            continue
            
        img_path = image_map[img_name]
        
        # Extract RLEs for this image
        img_df = df[df['image_name'] == img_name]
        rle_dict = {}
        for _, row in img_df.iterrows():
            rle_dict[row['class']] = row['rle']
            
        tasks.append((img_name, img_path, rle_dict, classes))
    
    # Run Multiprocessing
    num_cores = min(8, multiprocessing.cpu_count())
    print(f"Running on {num_cores} cores...")
    
    all_results = []
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use imap to get results as they complete
        # Chunksize > 1 might help performance slightly but default is fine
        results_iter = pool.imap(process_one_image_wrapper, tasks, chunksize=1)
        
        for res in tqdm(results_iter, total=len(tasks), desc="Processing"):
            all_results.extend(res)
            
    # Save
    print("Saving results...")
    new_df = pd.DataFrame(all_results)
    
    # Ensure column order
    if 'image_name' in new_df.columns:
         new_df = new_df[['image_name', 'class', 'rle']]
         
    new_df.to_csv(save_path, index=False)
    print(f"Saved processed CSV to {save_path}")

if __name__ == "__main__":
    # =========================================================
    # [사용자 설정] 아래 경로를 본인의 환경에 맞게 설정하세요
    INPUT_CSV = "submission_WJH_073_ensemble_sliding_Gaussian.csv" # 입력 CSV (앙상블 결과 등)
    OUTPUT_CSV = "submission_postprocessed.csv"                 # 저장될 경로
    
    IMAGE_ROOT = Config.TEST_IMAGE_ROOT          # 테스트 이미지 폴더
    # =========================================================

    print(f">> Input: {INPUT_CSV}")
    print(f">> Output: {OUTPUT_CSV}")
    
    process_csv(INPUT_CSV, OUTPUT_CSV, IMAGE_ROOT)
