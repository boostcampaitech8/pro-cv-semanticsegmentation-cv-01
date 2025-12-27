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

def get_probs(model, loader, save_dir=None, return_downsampled=None, tta_mode=None, tta_scales=None):
    """
    Modular function for TTA Inference.
    
    Modes:
    1. Standalone (Default): Returns dict {name: RLE_String}
    2. Ensemble Phase 1 (Optimization): Set return_downsampled=256 -> Returns dict {name: LowRes_Prob_Array}
    3. Ensemble Phase 2 (Test): Set save_dir="path" -> Saves HighRes_Float16_NPY, Returns None
    """
    # Defaults from Config if not provided
    if tta_mode is None: tta_mode = getattr(Config, 'TTA_MODE', '')
    if tta_scales is None: tta_scales = getattr(Config, 'TTA_SCALES', [1.0])
    
    # Ensure directory exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    print(f">> [get_probs] Mode='{tta_mode}', Scales={tta_scales}")
    if save_dir: print(f"   -> Saving results to {save_dir} (float16)")
    if return_downsampled: print(f"   -> Returning downsampled results (Size: {return_downsampled})")
    
    results = {} 
    model.eval()
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Inference"):
            images = images.cuda()
            base_h, base_w = images.shape[2], images.shape[3]
            
            # --- Optimized TTA Loop ---
            ensemble_pred_cpu = None 
            count = 0
            
            for scale in tta_scales:
                # 1. Resize Input
                if scale != 1.0:
                    new_h = int(base_h * scale)
                    new_w = int(base_w * scale)
                    pad_h = (32 - new_h % 32) % 32
                    pad_w = (32 - new_w % 32) % 32
                    target_h = new_h + pad_h
                    target_w = new_w + pad_w
                    
                    scaled_images = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
                else:
                    scaled_images = images
                
                # 2. Augment
                inputs = [scaled_images]
                if 'hflip' in tta_mode:
                    inputs.append(torch.flip(scaled_images, dims=[3]))
                if 'vflip' in tta_mode:
                    inputs.append(torch.flip(scaled_images, dims=[2]))
                
                batch_inputs = torch.cat(inputs, dim=0) 
                
                # 3. Inference
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(batch_inputs)
                    if isinstance(outputs, dict): outputs = outputs['out']
                
                del batch_inputs, scaled_images, inputs
                
                # 4. De-augment
                current_idx = 1
                if 'hflip' in tta_mode:
                    outputs[current_idx] = torch.flip(outputs[current_idx], dims=[3])
                    current_idx += 1
                if 'vflip' in tta_mode:
                    outputs[current_idx] = torch.flip(outputs[current_idx], dims=[2])
                    current_idx += 1
                    
                # 5. Restore Size & Move to CPU
                if scale != 1.0:
                     outputs = F.interpolate(outputs, size=(base_h, base_w), mode='bilinear', align_corners=False)
                
                scale_sum = torch.sum(outputs, dim=0, keepdim=True).cpu() 
                
                if ensemble_pred_cpu is None:
                    ensemble_pred_cpu = scale_sum
                else:
                    ensemble_pred_cpu += scale_sum
                    
                count += outputs.shape[0]
                
                del outputs, scale_sum
                torch.cuda.empty_cache()
            
            # Average Logits (CPU)
            avg_logits = ensemble_pred_cpu / count
            
            # Canonical Size (2048)
            if avg_logits.shape[2] != 2048:
                avg_logits = F.interpolate(avg_logits, size=(2048, 2048), mode="bilinear", align_corners=False)
                
            fname = os.path.basename(image_names[0])

            # --- Output Handling ---
            
            # 1. Downsampled Return (Memory Efficient for Optimization)
            if return_downsampled:
                 # Resize to small size (e.g. 256)
                 small_logits = F.interpolate(avg_logits, size=(return_downsampled, return_downsampled), mode="bilinear", align_corners=False)
                 prob = torch.sigmoid(small_logits).squeeze(0).numpy() # (C, S, S)
                 results[fname] = prob
            
            # 2. Disk Save (Float16 NPY)
            if save_dir:
                prob_full = torch.sigmoid(avg_logits).squeeze(0).half().numpy() # Float16
                save_path = os.path.join(save_dir, fname + ".npy")
                np.save(save_path, prob_full)
                
            # 3. Default RLE Return (If not saving)
            if not save_dir and not return_downsampled:
                prob_map = torch.sigmoid(avg_logits[0]) 
                pred_mask = (prob_map > 0.5)
                for c, segm in enumerate(pred_mask):
                    rle = encode_mask_to_rle(segm.numpy())
                    class_name = Config.CLASSES[c]
                    results[f"{class_name}_{fname}"] = rle
            
            del avg_logits, ensemble_pred_cpu
                
    return results

def test():
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError: return
    
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Get RLE Dict directly (RAM Efficient)
    results_dict = predict_and_encode(model, test_loader)
    
    # Save CSV
    print("Saving CSV...")
    
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
