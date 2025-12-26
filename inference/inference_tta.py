import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import os
import importlib
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from utils import encode_mask_to_rle

def get_probs(model, loader, tta_mode=None, tta_scales=None):
    """
    Modular function to get probabilities using TTA.
    """
    # Defaults from Config if not provided
    if tta_mode is None: tta_mode = getattr(Config, 'TTA_MODE', 'hflip')
    if tta_scales is None: tta_scales = getattr(Config, 'TTA_SCALES', [1.0])
    
    print(f">> [inference_tta] Mode='{tta_mode}', Scales={tta_scales}")
    
    results = {} # Store probs directly? Or accumulate? 
    # For ensemble, we usually iterate loader inside ensemble script. 
    # But here we want to delegate the *inference logic* to this module.
    # So we loop efficiently.
    
    model.eval()
    accum_probs = {} # Key: image_name, Val: Prob Tensor (C, H, W)
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="TTA Inference"):
            images = images.cuda()
            base_h, base_w = images.shape[2], images.shape[3]
            
            # --- TTA Loop ---
            ensemble_pred = None
            count = 0
            
            for scale in tta_scales:
                # 1. Resize Input
                if scale != 1.0:
                    scaled_images = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    scaled_images = images
                
                # 2. Augment
                inputs = [scaled_images]
                if 'hflip' in tta_mode:
                    inputs.append(torch.flip(scaled_images, dims=[3]))
                if 'vflip' in tta_mode:
                    inputs.append(torch.flip(scaled_images, dims=[2]))
                
                batch_inputs = torch.cat(inputs, dim=0) # (B, 3, H, W)
                
                # 3. Model
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(batch_inputs)
                    if isinstance(outputs, dict): outputs = outputs['out']
                
                # 4. De-augment
                current_idx = 1
                if 'hflip' in tta_mode:
                    outputs[current_idx] = torch.flip(outputs[current_idx], dims=[3])
                    current_idx += 1
                if 'vflip' in tta_mode:
                    outputs[current_idx] = torch.flip(outputs[current_idx], dims=[2])
                    current_idx += 1
                    
                # 5. Restore Size
                if scale != 1.0:
                    outputs = F.interpolate(outputs, size=(base_h, base_w), mode='bilinear', align_corners=False)
                
                scale_sum = torch.sum(outputs, dim=0, keepdim=True) # (1, C, H, W)
                if ensemble_pred is None: ensemble_pred = scale_sum
                else: ensemble_pred += scale_sum
                count += outputs.shape[0]
            
            # Average Logits
            avg_logits = ensemble_pred / count
            
            # Resize to 2048 (Canonical Size) for Ensemble Consistency
            if avg_logits.shape[2] != 2048:
                avg_logits = F.interpolate(avg_logits, size=(2048, 2048), mode="bilinear", align_corners=False)
                
            prob = torch.sigmoid(avg_logits).cpu().numpy() # (1, C, H, W)
            
            # Store
            for p, name in zip(prob, image_names):
                accum_probs[os.path.basename(name)] = p

    return accum_probs

def test():
    # ... (Legacy test function wrapper) ...
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError: return
    
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Call modular function
    probs_dict = get_probs(model, test_loader)
    
    # Post-process
    results_dict = {}
    for name, prob in probs_dict.items():
        # prob: (C, H, W)
        pred_mask = (prob > 0.5)
        for c, segm in enumerate(pred_mask):
            rle = encode_mask_to_rle(segm)
            class_name = Config.CLASSES[c]
            results_dict[f"{class_name}_{name}"] = rle

    # ... Save CSV (abbreviated) ...
    print("Saving CSV...")
    # (Existing CSV saving logic)
    
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        sample_df.to_csv(f"submission_{Config.EXPERIMENT_NAME}_TTA.csv", index=False)

if __name__ == '__main__':
    test()
