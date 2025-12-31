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

def predict_one_image(model, image, window_size=None, stride=None, **kwargs):
    # Adapter for uniform interface
    if window_size is None: window_size = getattr(Config, 'WINDOW_SIZE', 1024)
    if stride is None: stride = getattr(Config, 'STRIDE', 512)
    
    # Ensure image is on CUDA (Phase 2 passes CPU tensor)
    image = image.cuda()
    
    return predict_sliding_image(model, image, window_size, stride)

def predict_sliding_image(model, image, window_size, stride):
    # Core Sliding Logic
    # image: (C, H, W) Tensor on CUDA
    _, h_img, w_img = image.shape
    
    patches = []
    positions = []
    
    for y in range(0, h_img - window_size + 1, stride):
        for x in range(0, w_img - window_size + 1, stride):
            patch = image[:, y:y+window_size, x:x+window_size]
            patches.append(patch)
            positions.append((y, x))
            
    if not patches: return None # Handle error
        
    patches = torch.stack(patches) # (N, 3, W, W)
    
    # Process in chunks to avoid OOM (e.g. 9 patches of 1024x1024 is too big)
    chunk_size = 2 
    outputs_list = []
    
    with torch.amp.autocast(device_type="cuda"):
        for i in range(0, patches.shape[0], chunk_size):
            batch_patches = patches[i : i + chunk_size]
            batch_out = model(batch_patches)
            if isinstance(batch_out, dict): batch_out = batch_out['out']
            outputs_list.append(batch_out)
            
    outputs = torch.cat(outputs_list, dim=0)
    
    # Keep as logits or probs? Probs is safer for accumulation
    pred_patches = torch.sigmoid(outputs) 
    
    num_classes = pred_patches.shape[1]
    full_mask = torch.zeros((num_classes, h_img, w_img), device=image.device)
    count_mask = torch.zeros((num_classes, h_img, w_img), device=image.device)
    
    for (y, x), patch in zip(positions, pred_patches):
        full_mask[:, y:y+window_size, x:x+window_size] += patch
        count_mask[:, y:y+window_size, x:x+window_size] += 1
        
    count_mask[count_mask == 0] = 1
    full_mask /= count_mask
    
    return full_mask

def get_probs(model, loader, save_dir=None, return_downsampled=None, window_size=None, stride=None, save_dtype='float16', **kwargs):
    if window_size is None: window_size = getattr(Config, 'WINDOW_SIZE', 1024)
    if stride is None: stride = getattr(Config, 'STRIDE', 512)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    print(f">> [inference_sliding] Window={window_size}, Stride={stride}")
    if save_dir: print(f"   -> Saving results to {save_dir} (float16)")
    if return_downsampled: print(f"   -> Returning downsampled results (Size: {return_downsampled})")
    
    results = {}
    model.eval()
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Sliding Inference"):
            # images: (B, 3, H, W) - usually B=1
            for img, name in zip(images, image_names):
                img = img.cuda()
                # prob is (C, H, W) Tensor on CUDA
                prob = predict_sliding_image(model, img, window_size, stride)
                
                fname = os.path.basename(name)
                
                # 1. Downsampled Return
                if return_downsampled:
                    # Resize to target size if needed
                    if isinstance(return_downsampled, int):
                        target_size = (return_downsampled, return_downsampled)
                    else:
                        target_size = return_downsampled
                        
                    if prob.shape[-2:] != target_size:
                        small_prob = F.interpolate(prob.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
                    else:
                        small_prob = prob

                    # [MODIFIED] Disk Save Strategy (OOM Fix)
                    if save_dir:
                        # If save_dir is present, save to disk and DO NOT return (save RAM)
                        # Option to save as uint8 via an external flag or implicit assumption
                        # Checking if 'small_prob' implies optimization usage
                        
                        # Assuming optimization mode wants uint8 to save disk space as well
                        prob_np = (small_prob.cpu().numpy() * 255).astype(np.uint8)
                        np.save(os.path.join(save_dir, fname + ".npy"), prob_np)
                        
                    else:
                        # Optimization: Store as uint8 to save 4x RAM
                        results[fname] = (small_prob.cpu().numpy() * 255).astype(np.uint8)
                        
                # 2. Disk Save (Float16) - Only if NOT downsampled (Legacy mode)
                elif save_dir:
                    if save_dtype == 'uint8':
                        # Optimize Memory: uint8 (0-255)
                        prob_np = (prob.cpu().numpy() * 255).astype(np.uint8)
                        np.save(os.path.join(save_dir, fname + ".npy"), prob_np)
                    else:
                        prob_np = prob.half().cpu().numpy()
                        np.save(os.path.join(save_dir, fname + ".npy"), prob_np)
                    
                # 3. Default: RLE Encoding (RAM Safe)
                else:
                    # Resize to 2048 if needed
                    if prob.shape[1] != 2048:
                        prob = F.interpolate(prob.unsqueeze(0), size=(2048, 2048), mode="bilinear", align_corners=False).squeeze(0)
                    
                    pred_mask = (prob > 0.5)
                    for c, segm in enumerate(pred_mask):
                        rle = encode_mask_to_rle(segm.cpu().numpy())
                        class_name = Config.CLASSES[c]
                        results[f"{class_name}_{fname}"] = rle
                
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
    
    # Get RLE Dict directly
    results_dict = get_probs(model, test_loader)
    
    # Save CSV

    # ... Save CSV ...
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        sample_df.to_csv(f"submission_{Config.EXPERIMENT_NAME}_Sliding.csv", index=False)

if __name__ == '__main__':
    test()
