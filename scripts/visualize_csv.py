```python
import sys
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import Config
from utils import decode_rle_to_mask

def visualize_csv(csv_path, image_root, save_dir, num_samples=5, target_image=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    image_names = df['image_name'].unique()
    print(f"Total images in CSV: {len(image_names)}")
    
    if target_image:
        if target_image in image_names:
            selected_images = [target_image]
        else:
            print(f"Image {target_image} not found in CSV.")
            return
    else:
        selected_images = random.sample(list(image_names), min(num_samples, len(image_names)))
        
    # Color palette (Random)
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(Config.CLASSES), 3), dtype=np.uint8)
    
    for img_name in selected_images:
        print(f"Visualizing {img_name}...")
        
        # Determine image path
        # Recursively find image
        img_path = None
        for root, dirs, files in os.walk(image_root):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break
        
        if img_path is None:
            print(f"Image file not found for {img_name}")
            continue
            
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        # Reconstruct Mask
        img_df = df[df['image_name'] == img_name]
        
        overlay = image.copy()
        
        # Draw masks
        for _, row in img_df.iterrows():
            cls_name = row['class']
            rle = row['rle']
            
            if isinstance(rle, str) and len(rle) > 0:
                mask = decode_rle_to_mask(rle, height=H, width=W)
                
                # Get color
                cls_idx = Config.CLASS2IND[cls_name]
                color = colors[cls_idx].tolist()
                
                # Apply color to mask region
                # Alpha blending
                alpha = 0.4
                
                # Iterate channels for color
                for c in range(3):
                    overlay[:, :, c] = np.where(mask == 1, 
                                                overlay[:, :, c] * (1 - alpha) + color[c] * alpha, 
                                                overlay[:, :, c])
                    
                # Draw Contour for clarity
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)

        # Save
        save_path = os.path.join(save_dir, f"vis_{img_name}")
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"{img_name}")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--image_root", type=str, default=Config.TEST_IMAGE_ROOT, help="Image root directory")
    parser.add_argument("--save_dir", type=str, default="outputs/visualization", help="Directory to save images")
    parser.add_argument("--num", type=int, default=5, help="Number of random samples")
    parser.add_argument("--target", type=str, default=None, help="Specific image name to visualize")
    
    args = parser.parse_args()
    
    visualize_csv(args.csv, args.image_root, args.save_dir, args.num, args.target)
