import torch
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

def load_model(path):
    print(f">> Loading {path}...")
    model = torch.load(path, map_location='cuda', weights_only=False)
    model.eval()
    return model

# Reuse automated weight finding logic? 
# Yes, but for brevity/cleanliness, I'll rely on the 'weighted' strategy calling `find_ensemble_weights` tool logic if needed.
# Or just implement the inference part here. The user emphasized usage of inference_*.py files.
# Implementation of auto-weight finding using modular inference files is complex (needs to run all modules on val set).
# For now, I'll stick to 'manual' or simple 'weighted' (reusing simple resizing logic for weights, but using modular for final inference).

def parse_ensemble_config():
    raw_config = getattr(Config, 'ENSEMBLE_MODELS', [])
    parsed_config = []
    
    for item in raw_config:
        cfg = item.copy()
        if 'inference_file' not in cfg:
            # Default to basic TTA if not specified
            cfg['inference_file'] = 'inference.inference_tta'
        parsed_config.append(cfg)
    return parsed_config

def test():
    models_config = parse_ensemble_config()
    if not models_config:
        print("Please configure ENSEMBLE_MODELS in config.py")
        return

    # 1. Load Models & Weights (assuming manual or previously found weights)
    # If auto-weight is needed, it should ideally use the SAME modular logic on val set.
    # For now, let's assume 'manual' or equal weights to keep this script focused on the modular integration.
    # (Integrating modular val inference for optimization is a logical next step but might be overkill for this turn)
    
    weights = getattr(Config, 'ENSEMBLE_WEIGHTS', None)
    if weights is None: weights = [1.0/len(models_config)] * len(models_config)
    print(f">> Ensemble Weights: {weights}")

    # 2. Dataset
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
        XRayInferenceDataset = dataset_module.XRayInferenceDataset
        get_transforms = dataset_module.get_transforms
    except ModuleNotFoundError: return
        
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. Collect Probs from Modules
    all_model_probs = [] # List of dicts {filename: prob_array}
    
    for i, cfg in enumerate(models_config):
        model = load_model(cfg['path'])
        inf_module_name = cfg['inference_file']
        print(f">> [Model {i+1}] Using {inf_module_name}...")
        
        try:
            inf_module = importlib.import_module(inf_module_name)
        except ModuleNotFoundError:
            print(f"Error: Module {inf_module_name} not found.")
            return

        # Call the exposed API
        # Pass **cfg so it receives 'tta_mode', 'window_size', etc.
        probs = inf_module.get_probs(model, test_loader, **cfg)
        all_model_probs.append(probs)
        
        # Free memory
        del model
        torch.cuda.empty_cache()

    # 4. Ensemble Aggregation
    print(">> Aggregating Results...")
    results_dict = {}
    
    # Get all filenames
    filenames = list(all_model_probs[0].keys())
    
    for name in tqdm(filenames, desc="Ensemble Voting"):
        final_prob = None
        
        for i, model_probs in enumerate(all_model_probs):
            p = model_probs[name] # (C, H, W) numpy
            w = weights[i]
            
            if final_prob is None:
                final_prob = w * p
            else:
                final_prob += w * p
        
        # Threshold
        pred_mask = (final_prob > 0.5)
        
        # RLE
        for c, segm in enumerate(pred_mask):
            rle = encode_mask_to_rle(segm)
            class_name = Config.CLASSES[c]
            results_dict[f"{class_name}_{name}"] = rle

    # 5. Save
    print("Saving Ensemble CSV...")
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        sample_df.to_csv(f"submission_ensemble_modular.csv", index=False)
        print(f"Saved: submission_ensemble_modular.csv")

if __name__ == '__main__':
    test()
