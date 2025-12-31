import sys
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from utils import encode_mask_to_rle, decode_rle_to_mask
try:
    from inference.convcrf import ConvCRF
except ImportError:
    from convcrf import ConvCRF

class CSVDataset(Dataset):
    def __init__(self, df, image_root, classes):
        self.df = df
        self.image_root = image_root
        self.classes = classes
        self.image_names = df['image_name'].unique()
        
        # Pre-scan images
        print(f"Scanning images in {image_root}...")
        self.image_map = {}
        for root, dirs, files in os.walk(image_root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_map[f] = os.path.join(root, f)

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if img_name not in self.image_map:
            # Should not happen if data is clean
            print(f"Missing {img_name}")
            return None
        
        img_path = self.image_map[img_name]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Masks
        H, W, _ = image.shape
        img_df = self.df[self.df['image_name'] == img_name]
        
        masks = np.zeros((len(self.classes), H, W), dtype=np.float32)
        
        for _, row in img_df.iterrows():
            cls_name = row['class']
            rle = row['rle']
            if isinstance(rle, str) and len(rle) > 0:
                mask = decode_rle_to_mask(rle, height=H, width=W)
                cls_idx = Config.CLASS2IND[cls_name]
                masks[cls_idx] = mask
                
        # Return tensors
        # Image: (3, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # Masks: (C, H, W)
        masks_tensor = torch.from_numpy(masks).float()
        
        return img_name, image_tensor, masks_tensor

def process_convcrf():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    INPUT_CSV = "outputs/WJH_054_ensemble3.csv"
    OUTPUT_CSV = "outputs/WJH_060_ensemble3_convcrf.csv"
    IMAGE_ROOT = Config.TEST_IMAGE_ROOT
    
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    classes = Config.CLASSES
    
    dataset = CSVDataset(df, IMAGE_ROOT, classes)
    
    # Init CRF model
    # We do binary classification for each channel
    # filter_size=7 (3 neighbors each side)
    # Optimized params: sxy_bilateral=10 (Moderate), srgb_bilateral=3, compat_bilateral=6 (Moderate)
    crf = ConvCRF(num_classes=2, filter_size=7, n_iters=5,
                  sxy_bilateral=10, srgb_bilateral=3, compat_bilateral=6).to(device)
    
    results = []
    
    print("Starting processing...")
    # No batching for images because size is large and we iterate channels
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if data is None: continue
        
        img_name, image, masks = data
        image = image.to(device) # (3, H, W)
        masks = masks.to(device) # (C, H, W)
        
        C, H, W = masks.shape
        
        refined_masks_list = []
        
        # Process patches to save memory?
        # A single channel unroll (2 classes * 49 * 4M) ~ 1.6 GB. Safe.
        # We can enable cudnn benchmark
        
        with torch.no_grad():
            # Add batch dim for image: (1, 3, H, W)
            image_batch = image.unsqueeze(0)
            
            for c in range(C):
                mask = masks[c] # (H, W)
                
                # Optimization: Skip empty
                if mask.max() < 0.1:
                    refined_masks_list.append(torch.zeros_like(mask).byte())
                    continue
                
                # Create Binary Unary: (1, 2, H, W)
                # mask is [0, 1].
                # Prob(Fg) = 0.95/0.05 logic
                # Unary = -log(P)
                # P(Bg): 1 - mask (if mask=1->0, if mask=0->1) ? 
                # Relaxed: P(Fg) = mask*0.9 + 0.05
                
                prob_fg = mask * 0.9 + 0.05
                prob_bg = 1.0 - prob_fg
                
                probs = torch.stack([prob_bg, prob_fg], dim=0).unsqueeze(0) # (1, 2, H, W)
                unary = -torch.log(probs + 1e-6)
                
                # Run CRF
                q = crf(unary, image_batch) # (1, 2, H, W)
                
                # Get Fg prob
                res = q[0, 1] # (H, W)
                refined_masks_list.append((res > 0.5).byte())
                
        # Encode results
        refined_masks = torch.stack(refined_masks_list) # (C, H, W)
        refined_masks = refined_masks.cpu().numpy()
        
        for c, cls_name in enumerate(classes):
            rmask = refined_masks[c]
            if rmask.sum() > 0:
                rle = encode_mask_to_rle(rmask)
            else:
                rle = ""
                
            results.append({
                "image_name": img_name,
                "class": cls_name,
                "rle": rle
            })
            
    # Save
    new_df = pd.DataFrame(results)
    new_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_convcrf()
