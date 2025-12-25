
import os
import cv2
import glob
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def main():
    print(">>> Starting Pre-processing: PNG -> JPEG (PURE / NO CLAHE)")
    
    src_root = Config.IMAGE_ROOT
    dst_root = src_root.rstrip('/') + "_jpeg"
    
    os.makedirs(dst_root, exist_ok=True)
    print(f"Source: {src_root}")
    print(f"Dest:   {dst_root}")
    
    png_files = glob.glob(os.path.join(src_root, "**", "*.png"), recursive=True)
    print(f"Found {len(png_files)} images.")
    
    for p in tqdm(png_files):
        rel_path = os.path.relpath(p, src_root)
        dst_path = os.path.join(dst_root, rel_path)
        dst_path = os.path.splitext(dst_path)[0] + ".jpg"
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        img = cv2.imread(p)
        if img is None: continue
        
        # Save as JPEG (Quality 95)
        # Note: input is BGR, imwrite expects BGR. 
        cv2.imwrite(dst_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
    print(">>> Done.")

if __name__ == "__main__":
    main()
