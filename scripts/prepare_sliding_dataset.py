
import os
import cv2
import json
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import albumentations as A

# [CONFIG]
# [CONFIG]
DATA_ROOT = "../data" 
IMAGE_ROOT = os.path.join(DATA_ROOT, "train/DCM")
LABEL_ROOT = os.path.join(DATA_ROOT, "train/outputs_json")

# Output Directories
# Output Directories
OUT_IMAGE_ROOT = os.path.join(DATA_ROOT, "train/DCM_CLAHE")
OUT_MASK_ROOT = os.path.join(DATA_ROOT, "train/masks_png")

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def process_single(args):
    img_path, label_path, out_img_path, out_mask_path = args
    
    # 1. Image Processing (CLAHE)
    image = cv2.imread(img_path)
    if image is None: return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE
    transform = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
    ])
    image = transform(image=image)["image"]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Save as BGR
    
    cv2.imwrite(out_img_path, image)
    
    # 2. Mask Generation
    label_shape = (2048, 2048) # Mask is single channel? No, Multi-channel or Class-index map?
    # Usually segmentation uses class index map (0..29) for saving space
    # BUT current pipeline expects Multi-channel mask (H, W, C) or (C, H, W).
    # Saving 29-channel PNG is hard. 
    # Better to save as Single Channel (Index Map) and Expand at runtime?
    # Or save as .npy? .npy is fast but large.
    # Let's save as Single Channel Grayscale PNG (0-29).
    
    mask = np.zeros(label_shape, dtype=np.uint8)
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        # Draw masks
        # Note: If overlaps exist, later class overwrites earlier. 
        # Hand bones usually don't overlap much.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c] + 1 # 0 is background, 1..29 are classes
            points = np.array(ann["points"], dtype=np.int32)
            
            cv2.fillPoly(mask, [points], int(class_ind))
            
    cv2.imwrite(out_mask_path, mask)

def main():
    if not os.path.exists(OUT_IMAGE_ROOT): os.makedirs(OUT_IMAGE_ROOT)
    if not os.path.exists(OUT_MASK_ROOT): os.makedirs(OUT_MASK_ROOT)
    
    pngs = sorted(glob.glob(os.path.join(IMAGE_ROOT, "**/*.png"), recursive=True))
    jsons = sorted(glob.glob(os.path.join(LABEL_ROOT, "**/*.json"), recursive=True))
    
    print(f"Found {len(pngs)} images and {len(jsons)} jsons.")
    
    basename_to_json = {os.path.splitext(os.path.basename(p))[0]: p for p in jsons}

    tasks = []
    for img_path in pngs:
        basename = os.path.basename(img_path)
        name_only = os.path.splitext(basename)[0]
        
        # Look up JSON
        label_path = basename_to_json.get(name_only)
        if not label_path:
            # Maybe skip or warn? For now continue (mask will be empty)
            pass
        
        out_img_path = os.path.join(OUT_IMAGE_ROOT, name_only + ".jpg") # Save as JPG for space
        out_mask_path = os.path.join(OUT_MASK_ROOT, name_only + ".png") # Mask as PNG (Lossless)
        
        tasks.append((img_path, label_path, out_img_path, out_mask_path))
        
    print("Starting processing...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single, tasks), total=len(tasks)))
        
    print("Done! Artifacts saved to:")
    print(f" - Images: {OUT_IMAGE_ROOT}")
    print(f" - Masks:  {OUT_MASK_ROOT}")

if __name__ == "__main__":
    main()
