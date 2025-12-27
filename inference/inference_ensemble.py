import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import os
import importlib
import numpy as np
import sys
from scipy.optimize import minimize
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from utils import encode_mask_to_rle, dice_coef

def load_model(path):
    print(f">> Loading {path}...")
    model = torch.load(path, map_location='cuda', weights_only=False)
    model.eval()
    return model

def parse_ensemble_config():
    raw_config = getattr(Config, 'ENSEMBLE_MODELS', [])
    parsed_config = []
    
    for item in raw_config:
        cfg = item.copy()
        if 'inference_file' not in cfg:
            cfg['inference_file'] = 'inference.inference_tta'
        parsed_config.append(cfg)
    return parsed_config

def find_optimal_weights(models_probs, gt_masks):
    """
    models_probs: List of prob arrays (N, C, H, W) for each model (Valid Set)
    gt_masks: Ground Truth masks (N, C, H, W)
    """
    print(f">> Optimizing Weights for {len(models_probs)} models...")
    
    n_models = len(models_probs)
    initial_weights = [1.0 / n_models] * n_models

    def loss_func(weights):
        # Normalize weights
        weights = np.array(weights)
        weights = np.clip(weights, 0, 1) # Ensure non-negative
        if np.sum(weights) == 0: weights = initial_weights
        weights /= np.sum(weights) # Sum to 1
        
        # Weighted Ensemble Prob
        # Use a small subset or full set? Full set is fine for small val
        final_prob = np.zeros_like(models_probs[0])
        for i, w in enumerate(weights):
            final_prob += w * models_probs[i]
            
        pred_mask = (final_prob > 0.5).astype(np.float32)
        
        # Calculate -Dice (to minimize)
        # Assuming batch-mean dice
        score = 0
        eps = 1e-6
        
        # Vectorized Dice
        for c in range(gt_masks.shape[1]):
            input = pred_mask[:, c]
            target = gt_masks[:, c]
            intersection = (input * target).sum()
            union = input.sum() + target.sum() + eps
            score += (2. * intersection / union)
            
        score /= gt_masks.shape[1]
        return -score

    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * n_models
    
    res = minimize(loss_func, initial_weights, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-4)
    best_weights = res.x / np.sum(res.x)
    
    print(f">> Optimization Done. Best Dice: {-res.fun:.4f}")
    print(f">> Best Weights: {best_weights}")
    
    return best_weights

def test():
    models_config = parse_ensemble_config()
    if not models_config:
        print("Please configure ENSEMBLE_MODELS in config.py")
        return

class ValidationWrapper:
    """
    Adapts a Validation Loader (yielding image, mask) 
    to an Inference Loader signature (yielding image, image_names)
    by injecting filenames from a provided list.
    """
    def __init__(self, loader, filenames):
        self.loader = loader
        self.filenames = filenames
        self.batch_size = loader.batch_size if hasattr(loader, 'batch_size') else 1
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        iterator = iter(self.loader)
        for i, batch in enumerate(iterator):
            # Unpack (image, mask)
            if isinstance(batch, (list, tuple)):
                image = batch[0]
            elif isinstance(batch, dict):
                image = batch['image']
            else:
                image = batch
            
            # Get Names
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.filenames))
            batch_names = self.filenames[start_idx : end_idx]
            
            yield image, batch_names

def get_validation_data(dataset_module_path):
    """
    Helper to get validation loader and filenames transparently 
    supporting both Standard and DALI datasets.
    """
    mod = importlib.import_module(dataset_module_path)
    
    if hasattr(mod, 'get_dali_loader'):
        # DALI Case
        loader = mod.get_dali_loader(is_train=False, batch_size=1)
        # Access source filenames (Assume DALI wrapper structure)
        filenames = loader.source.filenames
    else:
        # Standard Case
        XRayDataset = mod.XRayDataset
        get_transforms = mod.get_transforms
        dataset = XRayDataset(is_train=False, transforms=get_transforms(is_train=False))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        filenames = dataset.filenames
        
    return loader, filenames

def test():
    models_config = parse_ensemble_config()
    if not models_config:
        print("Please configure ENSEMBLE_MODELS in config.py")
        return

    # Base module for types - mostly unused now due to dynamic loading below
    # but kept for XRayInferenceDataset usage in Phase 2
    global_dataset_module = importlib.import_module(Config.DATASET_FILE)

    # ====================================================
    # 1. Weight Optimization (Validation Step)
    # ====================================================
    USE_OPTIMIZATION = getattr(Config, 'ENSEMBLE_USE_OPTIMIZATION', False)
    weights = getattr(Config, 'ENSEMBLE_WEIGHTS', None)

    if USE_OPTIMIZATION:
        print("\n[Phase 1] Auto-Finding Ensemble Weights on Validation Set...")
        
        # Load Reference Validation Data (Target GT)
        # Use Config.DATASET_FILE as the source of truth for GT
        valid_loader, valid_filenames = get_validation_data(Config.DATASET_FILE)
        
        # Collect Valid Probs and GT
        valid_model_probs = [] # [Model1_Probs(N,C,H,W), Model2_Probs...]
        valid_gts = []
        
        # Load GT once
        print("Loading GT Masks...")
        for batch in valid_loader:
            # Handle DALI/Standard Return (image, mask)
            if isinstance(batch, (list, tuple)):
                masks = batch[1]
            elif isinstance(batch, dict):
                masks = batch['mask']
            
            # Ensure CPU Numpy
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
                
            valid_gts.append(masks)
            
        valid_gts = np.concatenate(valid_gts, axis=0) # (N, C, H, W)
        
        # Run Inference for each model on Valid
        for i, cfg in enumerate(models_config):
            model = load_model(cfg['path'])
            inf_module_name = cfg['inference_file']
            
            # [NEW] Determine Dataset File for this model
            target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
            print(f">> [Valid Inference] Model {i+1} ({inf_module_name}) | Dataset: {target_dataset_file}...")
            
            try:
                # Dynamic Dataset Loading (Validation Adapter)
                curr_valid_loader, curr_valid_filenames = get_validation_data(target_dataset_file)
                
                # Wrap for Inference (image, mask) -> (image, filename)
                inference_adapter = ValidationWrapper(curr_valid_loader, curr_valid_filenames)
                
                inf_module = importlib.import_module(inf_module_name)
                
                # Check if get_probs supports array return or we construct it
                probs_dict = inf_module.get_probs(model, inference_adapter, **cfg)
                
                # Align with loader (Use global valid_dataset for canonical filename order)
                # This assumes filename lists match across datasets
                aligned_probs = []
                for fname in valid_filenames: 
                     aligned_probs.append(probs_dict[os.path.basename(fname)])
                
                valid_model_probs.append(np.stack(aligned_probs))
                
            except Exception as e:
                print(f"Error during optimization: {e}")
                # return # Do not return, maybe just skip optimization?
                # Actually if optimization fails, we should probably fallback to avg
                print("Fallback to Average weights due to error.")
                weights = None
                break
            
            del model, probs_dict
            torch.cuda.empty_cache()

        # Find Weights if successful
        if weights is None and len(valid_model_probs) == len(models_config):
             weights = find_optimal_weights(valid_model_probs, valid_gts)
        
        # Clean up Valid memory
        if 'valid_model_probs' in locals(): del valid_model_probs
        if 'valid_gts' in locals(): del valid_gts
        import gc; gc.collect()
        
    elif weights is None:
        weights = [1.0/len(models_config)] * len(models_config)
    
    print(f"\n>> Final Ensemble Weights: {weights}")

    # ====================================================
    # 2. Test Inference (Main Step)
    # ====================================================
    # ====================================================
    # 2. Test Inference (Main Step)
    # ====================================================
    print("\n[Phase 2] Running Inference on Test Set...")
    
    # Use global dataset module for the base test loader structure if needed
    # But usually we re-instantiate inside the loop. 
    # However, 'test_dataset' init here is mainly for printing or structure? 
    # Actually checking original code: lines 175-176 do instantiate a loader but it's UNUSED!
    # Because inside the loop (line 196) we re-instantiate `test_dataset` and `test_loader` for every model.
    # So we can just remove the unused instantiation or fix it.
    # Let's keep it but fix the name reference.
    
    XRayInferenceDataset = global_dataset_module.XRayInferenceDataset
    get_transforms = global_dataset_module.get_transforms
    
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    
    # Run Inference for each model on Test
    all_model_probs = [] # List of dicts {filename: prob}
    
    for i, cfg in enumerate(models_config):
        model = load_model(cfg['path'])
        inf_module_name = cfg['inference_file']
        
        # [NEW] Determine Dataset File for this model
        target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
        print(f">> [Test Inference] Model {i+1} ({inf_module_name}) | Dataset: {target_dataset_file}...")
        
        try:
            # Dynamic Dataset Loading
            ds_module = importlib.import_module(target_dataset_file)
            get_transforms_fn = ds_module.get_transforms
            XRayInferenceDataset_cls = ds_module.XRayInferenceDataset
            
            # Re-instantiate Loader with specific transforms
            test_dataset = XRayInferenceDataset_cls(transforms=get_transforms_fn(is_train=False))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
            
            inf_module = importlib.import_module(inf_module_name)
            probs = inf_module.get_probs(model, test_loader, **cfg)
            all_model_probs.append(probs)
            
        except Exception as e:
             print(f"Error processing model {i+1}: {e}")
             return

        del model
        torch.cuda.empty_cache()

    # Aggregation
    print(">> Aggregating Results...")
    results_dict = {}
    filenames = list(all_model_probs[0].keys())
    
    for name in tqdm(filenames, desc="Ensemble Voting"):
        final_prob = None
        for i, model_probs in enumerate(all_model_probs):
            p = model_probs[name]
            w = weights[i]
            if final_prob is None: final_prob = w * p
            else: final_prob += w * p
        
        pred_mask = (final_prob > 0.5)
        for c, segm in enumerate(pred_mask):
            rle = encode_mask_to_rle(segm)
            class_name = Config.CLASSES[c]
            results_dict[f"{class_name}_{name}"] = rle

    # Save
    print("Saving Ensemble CSV...")
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
        sample_df['rle'] = final_rles
        
        num_models = len(models_config)
        
        if USE_OPTIMIZATION:
            strategy = "opt"
        elif getattr(Config, 'ENSEMBLE_WEIGHTS', None) is not None:
            strategy = "manual"
        else:
            strategy = "avg"
            
        # Format weights strictly to 2 decimal places, joined by hyphen
        weights_str = "-".join([f"{w:.2f}" for w in weights])
        
        save_name = f"submission_ens_{num_models}models_{strategy}_{weights_str}.csv"
        sample_df.to_csv(save_name, index=False)
        print(f"Saved: {save_name}")

if __name__ == '__main__':
    test()
