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

def get_probs(model, loader, **kwargs):
    """
    Standard inference returning probabilities. 
    Accepts arbitrary kwargs to be compatible with ensemble calls but ignores them.
    """
    print(f">> [inference] Standard Inference")
    
    accum_probs = {}
    model.eval()
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Inference"):
            images = images.cuda()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
            
            # Resize
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs).cpu().numpy()
            
            for output, image_path in zip(outputs, image_names):
                image_name = os.path.basename(image_path)
                accum_probs[image_name] = output
                
    return accum_probs

def test():
    # Legacy wrapper
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError: return
    
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    # Fix: Handle file renaming/missing issue gracefully
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    print(f"Loading Model from {model_path}")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()
    
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    results_dict = {}
    
    print("Start Inference (Memory Efficient Mode)...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Inference"):
            images = images.cuda()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
            
            # Resize
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).detach().cpu().numpy() # Boolean array (smaller than float)
            
            for output, image_path in zip(outputs, image_names):
                image_name = os.path.basename(image_path)
                
                # Encode RLE immediately and discard the array
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    class_name = Config.CLASSES[c]
                    results_dict[f"{class_name}_{image_name}"] = rle
    
    # Save CSV
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        save_path = f"submission_{Config.EXPERIMENT_NAME}.csv"
        sample_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    else:
        print("sample_submission.csv not found.")

if __name__ == '__main__':
    test()