import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import importlib
import numpy as np
import itertools
from config import Config

def find_best_tta_combination():
    # 1. Setup
    print(">> [TTA Combination Search] Initializing...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Dataset Module
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
        get_dali_loader = dataset_module.get_dali_loader
    except (ImportError, AttributeError):
        print(f"Error: Could not load 'get_dali_loader' from {Config.DATASET_FILE}")
        return

    # Load Model Module
    try:
        model_module = importlib.import_module(Config.MODEL_FILE)
        get_model = model_module.get_model
    except (ImportError, AttributeError):
        print(f"Error: Could not load 'get_model' from {Config.MODEL_FILE}")
        return

    # 2. Data Loader (Validation)
    print(f">> Loading Validation Set from {Config.DATASET_FILE}...")
    valid_loader = get_dali_loader(is_train=False, batch_size=1) 
    
    # 3. Model Load
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    print(f">> Loading Model from {model_path}...")
    
    # weights_only=False fix included
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    
    # 4. Define Candidate Scales (Full Set)
    small_scales = [0.5, 0.75, 0.8, 0.9]
    base_scales = [1.0]
    large_scales = [1.1, 1.25, 1.5]
    
    all_scales = sorted(list(set(small_scales + base_scales + large_scales)))
    print(f">> Candidate Scales: {all_scales}")
    print(f">> Search Space: Small={small_scales}, Base={base_scales}, Large={large_scales}")
    
    # Generate all valid combinations (One from each group: Small + Base + Large)
    combinations = list(itertools.product(small_scales, base_scales, large_scales))
    print(f">> Testing {len(combinations)} Combinations...\n")
    
    # Store accumulated dice scores for each combination
    comb_results = {comb: [] for comb in combinations}
    
    # 5. Iterate Validation Set
    print(">> Running Inference & Evaluation (Full GPU Mode)...")
    
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Evaluating")
        for batch_idx, (images, masks) in enumerate(pbar):
            # Input Handling
            if images.device.type != 'cuda': images = images.cuda()
            if masks.device.type != 'cuda': masks = masks.cuda()
            
            # Ground Truth Preparation
            target_h, target_w = masks.shape[2], masks.shape[3]
            y_true = masks.contiguous().flatten(2) # (B, C, H*W)

            # Store predictions for THIS image for ALL scales
            cached_outputs = {}
            
            # 1. Run inference for all candidate scales
            for scale in all_scales:
                # Resize Input with padding to multiple of 32 (HRNet Requirement)
                if scale != 1.0:
                    h, w = images.shape[2], images.shape[3]
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    
                    # Round to nearest multiple of 32
                    new_h = int(round(new_h / 32) * 32)
                    new_w = int(round(new_w / 32) * 32)
                    
                    scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                else:
                    scaled_images = images
                
                # Inference
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(scaled_images)
                    if isinstance(outputs, dict): outputs = outputs['out']
                
                # Restore Size
                if outputs.shape[2:] != (target_h, target_w):
                    outputs = F.interpolate(outputs, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # Cache on GPU (Faster, assuming memory is available)
                cached_outputs[scale] = outputs
                
            # 2. Evaluate all combinations for this image
            # Combination = Average of Logits
            for comb in combinations:
                # comb is tuple like (0.75, 1.0, 1.25)
                
                # Sum logits
                ensemble_logits = None
                for s in comb:
                    if ensemble_logits is None:
                        ensemble_logits = cached_outputs[s]
                    else:
                        ensemble_logits = ensemble_logits + cached_outputs[s]
                
                avg_logits = ensemble_logits / len(comb)
                
                # Calculate Dice
                preds = torch.sigmoid(avg_logits)
                preds = (preds > 0.5).float()
                y_pred = preds.flatten(2)
                
                # Dice calculation
                intersection = (y_pred * y_true).sum(dim=2)
                union = y_pred.sum(dim=2) + y_true.sum(dim=2)
                dice_score = (2. * intersection + 1e-4) / (union + 1e-4)
                mean_dice = dice_score.mean().item()
                
                comb_results[comb].append(mean_dice)
            
            # Cleanup
            del cached_outputs
            # No aggressive empty_cache() to keep it fast, relying on normal allocator
            
    # 6. Aggregate Results
    print("\n" + "=" * 40)
    print(" 🏆 TTA Combination Results (Top 5)")
    print("=" * 40)
    
    final_scores = []
    for comb, scores in comb_results.items():
        avg_score = sum(scores) / len(scores)
        final_scores.append((comb, avg_score))
    
    # Sort by score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (comb, score) in enumerate(final_scores[:5]):
        print(f" Rank {i+1}: Scales {str(comb):<20} | Mean Dice: {score:.4f}")
        
    print("-" * 40)
    best_comb = final_scores[0][0]
    print(f"✅ Best Combination: {best_comb}")
    print("=" * 40)

if __name__ == "__main__":
    find_best_tta_combination()
