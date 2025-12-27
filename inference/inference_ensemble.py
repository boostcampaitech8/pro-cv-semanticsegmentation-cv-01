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

    # Base module
    global_dataset_module = importlib.import_module(Config.DATASET_FILE)

    # ====================================================
    # 1. Weight Optimization (Validation Step)
    # ====================================================
    USE_OPTIMIZATION = getattr(Config, 'ENSEMBLE_USE_OPTIMIZATION', False)
    weights = getattr(Config, 'ENSEMBLE_WEIGHTS', None)

    if USE_OPTIMIZATION:
        print("\n[Phase 1] Auto-Finding Ensemble Weights on Validation Set (Low-Res Optimization)...")
        
        # 1. Load Reference Validation Data (Target GT)
        # Use Config.DATASET_FILE as the source of truth for GT
        valid_loader, valid_filenames = get_validation_data(Config.DATASET_FILE)
        
        print("Loading GT Masks (Downsampling to 256x256)...")
        gt_dict = {} # {filename: mask_array}
        
        # Accumulate GTs aligned with filenames
        # Note: valid_filenames is a list of all filenames in order
        # valid_loader yields single batches? No, batch_size=1 in get_validation_data
        
        current_idx = 0
        for batch in valid_loader:
            if isinstance(batch, (list, tuple)): masks = batch[1]
            elif isinstance(batch, dict): masks = batch['mask']
            
            if isinstance(masks, torch.Tensor):
                masks = F.interpolate(masks.float(), size=(256, 256), mode='nearest')
                masks = masks.cpu().numpy()
            
            # Batch size is 1, but technically could be more if changed later
            # let's iterate batch items
            batch_size = masks.shape[0]
            for b in range(batch_size):
                if current_idx < len(valid_filenames):
                    fname = os.path.basename(valid_filenames[current_idx])
                    gt_dict[fname] = masks[b]
                    current_idx += 1
        
        # 2. Collect Predictions from All Models
        model_probs_dicts = [] # List of {filename: prob}
        
        for i, cfg in enumerate(models_config):
            model = load_model(cfg['path'])
            inf_module_name = cfg['inference_file']
            target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
            print(f">> [Valid] Model {i+1} | Dataset: {target_dataset_file}...")
            
            try:
                curr_valid_loader, curr_valid_filenames = get_validation_data(target_dataset_file)
                inference_adapter = ValidationWrapper(curr_valid_loader, curr_valid_filenames)
                inf_module = importlib.import_module(inf_module_name)
                
                # Request Downsampled Probs
                probs_dict = inf_module.get_probs(model, inference_adapter, return_downsampled=256, **cfg)
                
                # Normalize keys to basename just in case
                normalized_probs_dict = {os.path.basename(k): v for k, v in probs_dict.items()}
                model_probs_dicts.append(normalized_probs_dict)
                
            except Exception as e:
                print(f"Error during optimization for Model {i+1}: {e}")
                print("Fallback to Average weights due to critical error.")
                weights = None
                model_probs_dicts = None
                break
            
            del model
            torch.cuda.empty_cache()
            
        # 3. Intersection & Alignment
        if model_probs_dicts:
            # Start with GT keys
            common_keys = set(gt_dict.keys())
            
            # Intersect with all models
            for idx, d in enumerate(model_probs_dicts):
                missing = common_keys - set(d.keys())
                if missing:
                    print(f"   [Warning] Model {idx+1} is missing {len(missing)} validation files (likely due to 'exclude'). Dropping them from optimization.")
                common_keys &= set(d.keys())
            
            sorted_keys = sorted(list(common_keys))
            print(f">> Optimized Intersection: {len(sorted_keys)} files (Original Ref: {len(gt_dict)})")
            
            if len(sorted_keys) == 0:
                 print("Error: No common files found between models and ground truth!")
                 weights = None
            else:
                 # Construct Aligned Arrays
                 valid_gts = np.stack([gt_dict[k] for k in sorted_keys])
                 valid_model_probs = []
                 for d in model_probs_dicts:
                     valid_model_probs.append(np.stack([d[k] for k in sorted_keys]))
                 
                 # Find Weights
                 weights = find_optimal_weights(valid_model_probs, valid_gts)
        
        import gc; gc.collect()
        
    elif weights is None:
        weights = [1.0/len(models_config)] * len(models_config)
    
    print(f"\n>> Final Ensemble Weights: {weights}")

    # ====================================================
    # 2. Test Inference (Disk Caching Strategy)
    # ====================================================
    print("\n[Phase 2] Running Inference on Test Set (Disk Caching Mode)...")
    
    # Setup Temp Dirs
    import shutil
    temp_root = "temp_ensemble_cache"
    if os.path.exists(temp_root): shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)
    
    # 1. Generate Probs to Disk
    filenames = None # Will grab from first model
    
    for i, cfg in enumerate(models_config):
        model = load_model(cfg['path'])
        inf_module_name = cfg['inference_file']
        target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
        
        # Create dedicated dir for this model
        model_save_dir = os.path.join(temp_root, f"model_{i}")
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f">> [Test] Model {i+1} -> Saving to {model_save_dir}...")
        
        try:
            ds_module = importlib.import_module(target_dataset_file)
            XRayInferenceDataset_cls = ds_module.XRayInferenceDataset
            get_transforms_fn = ds_module.get_transforms
            
            test_dataset = XRayInferenceDataset_cls(transforms=get_transforms_fn(is_train=False))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
            
            # Capture filenames from dataset 
            if filenames is None:
                # Standardize filenames list (basenames)
                filenames = [os.path.basename(f) for f in test_dataset.filenames]
            
            inf_module = importlib.import_module(inf_module_name)
            # CALL get_probs with save_dir
            inf_module.get_probs(model, test_loader, save_dir=model_save_dir, **cfg)
            
        except Exception as e:
             print(f"Error processing model {i+1}: {e}")
             return

        del model
        torch.cuda.empty_cache()

    # 2. Aggregate from Disk
    print("\n>> Aggregating Results from Disk & Creating CSV...")
    results_dict = {}
    
    # Iterate over images
    for name in tqdm(filenames, desc="Disk Ensemble"):
        final_prob = None
        
        # Load each model's prediction
        for i in range(len(models_config)):
            npy_path = os.path.join(temp_root, f"model_{i}", name + ".npy")
            
            if not os.path.exists(npy_path):
                print(f"[Warning] Missing {name} for model {i}")
                continue
                
            # Load Float16, Convert to Float32 for precision math
            p = np.load(npy_path).astype(np.float32)
            w = weights[i]
            
            if final_prob is None: final_prob = w * p
            else: final_prob += w * p
            
            # Remove file to save space immediately? 
            # Safe to remove if we don't need it anymore.
            os.remove(npy_path)
            
        # Threshold & Encode
        if final_prob is not None:
            pred_mask = (final_prob > 0.5)
            for c, segm in enumerate(pred_mask):
                rle = encode_mask_to_rle(segm)
                class_name = Config.CLASSES[c]
                results_dict[f"{class_name}_{name}"] = rle

    # Cleanup Temp Dir
    shutil.rmtree(temp_root)

    # Save CSV
    print("Saving Ensemble CSV...")
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            rle = results_dict.get(key, "")
            final_rles.append(rle)
            
        sample_df['rle'] = final_rles
        
        num_models = len(models_config)
        weights_str = "-".join([f"{w:.2f}" for w in weights])
        strategy = "opt" if USE_OPTIMIZATION else ("manual" if getattr(Config, 'ENSEMBLE_WEIGHTS', None) else "avg")
        
        save_name = f"submission_ens_{num_models}models_{strategy}_{weights_str}.csv"
        sample_df.to_csv(save_name, index=False)
        print(f"Saved: {save_name}")

if __name__ == '__main__':
    test()
