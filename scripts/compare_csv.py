import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import decode_rle_to_mask

def compare_csv(csv1_path, csv2_path):
    print(f"Comparing:")
    print(f"  A: {csv1_path}")
    print(f"  B: {csv2_path}")
    
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    if len(df1) != len(df2):
        print(f"!! Warning: Row counts differ: {len(df1)} vs {len(df2)}")
        
    # Sort to ensure alignment
    df1 = df1.sort_values(by=['image_name', 'class']).reset_index(drop=True)
    df2 = df2.sort_values(by=['image_name', 'class']).reset_index(drop=True)
    
    total_masks = len(df1)
    changed_masks = 0
    total_added_pixels = 0
    total_removed_pixels = 0
    
    print(f"Analyzing {total_masks} masks...")
    
    # We can check simple string equality first for speed
    diff_indices = []
    
    for i in range(total_masks):
        rle1 = str(df1.loc[i, 'rle'])
        rle2 = str(df2.loc[i, 'rle'])
        
        if rle1 != rle2:
            diff_indices.append(i)
            
    changed_masks = len(diff_indices)
    print(f">> Need to decode {changed_masks} changed masks to calculate pixel diffs...")
    
    # Analyze differences on a subset if too many? No, do all.
    for i in diff_indices:
        rle1 = str(df1.loc[i, 'rle'])
        rle2 = str(df2.loc[i, 'rle'])
        
        # Handle NaN/empty
        if rle1 == 'nan': rle1 = ''
        if rle2 == 'nan': rle2 = ''
        
        mask1 = decode_rle_to_mask(rle1, height=2048, width=2048)
        mask2 = decode_rle_to_mask(rle2, height=2048, width=2048)
        
        # Pixels 1 -> 0 (Removed)
        removed = np.sum((mask1 == 1) & (mask2 == 0))
        
        # Pixels 0 -> 1 (Added)
        added = np.sum((mask1 == 0) & (mask2 == 1))
        
        total_removed_pixels += removed
        total_added_pixels += added
        
    print(f"\n[Comparison Result]")
    print(f"Total Masks: {total_masks}")
    print(f"Changed Masks: {changed_masks} ({changed_masks/total_masks*100:.2f}%)")
    print(f"Total Pixels Added (Closing fills holes): {total_added_pixels}")
    print(f"Total Pixels Removed (Noise removal?): {total_removed_pixels}")
    
    net_change = total_added_pixels - total_removed_pixels
    print(f"Net Pixel Change: {net_change:+} pixels")
    
    if changed_masks > 0:
        avg_add_per_change = total_added_pixels / changed_masks
        avg_rem_per_change = total_removed_pixels / changed_masks
        print(f"Avg Added per changed mask: {avg_add_per_change:.1f}")
        print(f"Avg Removed per changed mask: {avg_rem_per_change:.1f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_csv.py <csv1> <csv2>")
    else:
        compare_csv(sys.argv[1], sys.argv[2])
