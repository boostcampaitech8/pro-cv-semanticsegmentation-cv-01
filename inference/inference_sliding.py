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
    
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(patches)
        if isinstance(outputs, dict): outputs = outputs['out']
    
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

def get_probs(model, loader, window_size=None, stride=None):
    if window_size is None: window_size = getattr(Config, 'WINDOW_SIZE', 1024)
    if stride is None: stride = getattr(Config, 'STRIDE', 512)
    
    print(f">> [inference_sliding] Window={window_size}, Stride={stride}")
    
    accum_probs = {}
    model.eval()
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Sliding Inference"):
            # images: (B, 3, H, W) - usually B=1
            for img, name in zip(images, image_names):
                img = img.cuda()
                prob = predict_sliding_image(model, img, window_size, stride)
                
                # Resize to 2048 if needed (sliding usually keeps full size though)
                if prob.shape[1] != 2048:
                    prob = F.interpolate(prob.unsqueeze(0), size=(2048, 2048), mode="bilinear", align_corners=False).squeeze(0)
                
                accum_probs[os.path.basename(name)] = prob.cpu().numpy()
                
    return accum_probs

def test():
    # Legacy wrapper
    # ... (Similar to inference_tta.py test() logic) ...
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError: return
    
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    probs_dict = get_probs(model, test_loader)
    
    results_dict = {}
    for name, prob in probs_dict.items():
        pred_mask = (prob > 0.5)
        for c, segm in enumerate(pred_mask):
            rle = encode_mask_to_rle(segm)
            class_name = Config.CLASSES[c]
            results_dict[f"{class_name}_{name}"] = rle

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
