import os
# Suppress Albumentations Update Check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Suppress Generic Warnings (DALI, etc)
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
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

def generate_simplex_grid(n_models, step=0.1):
    """
    Generates all valid combinations of weights (sum=1) with a given step size.
    Efficient recursive generator.
    """
    if n_models == 1:
        yield [1.0]
        return

    # For the last weight, it's determined by the rest, so we iterate n-1
    # Actually recursion is cleaner:
    # w1 can be 0, step, 2*step ... 1.0
    # w2 ...
    
    # Pre-calculate steps to avoid float errors
    n_steps = int(1.0 / step + 0.001)
    
    def internal_generate(current_weights, remaining_sum):
        depth = len(current_weights)
        if depth == n_models - 1:
            # Last one is fixed
            if remaining_sum >= -1e-9: # valid
                yield current_weights + [remaining_sum]
            return

        # Iterate possible steps for this depth
        # Max steps we can take is remaining_sum / step
        max_s = int(remaining_sum / step + 0.001)
        for s in range(max_s + 1):
            val = s * step
            internal_generate(current_weights + [val], remaining_sum - val)

    for w in internal_generate([], 1.0):
        yield w

def find_optimal_weights_robust(models_probs_list, gt_mask):
    """
    Robust Optimization: Grid Search -> Nelder-Mead on single channel.
    Input:
      models_probs_list: List of (N, H, W) arrays (Uint8 or float)
      gt_mask: (N, H, W) array (Uint8 or float)
    """
    n_models = len(models_probs_list)
    eps = 1e-6
    n_samples = gt_mask.shape[0]

    # --- OPTIMIZATION (GPU Accelerated + Batch-wise) ---
    # Fast & OOM Safe for Single Channel
    
    indices = np.arange(n_samples)
    
    # Enable Cudnn Benchmark
    torch.backends.cudnn.benchmark = True
    
    eps = 1e-6
    batch_size = 5 # Consistent with Global
    
    def loss_func(weights):
        # weights: (M,)
        # Convert to GPU Tensor
        w_tensor = torch.tensor(weights, device='cuda', dtype=torch.float32).view(-1, 1, 1, 1)
        
        total_inter = torch.tensor(0.0, device='cuda')
        total_union = torch.tensor(0.0, device='cuda')
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                b_idx = indices[start_idx:end_idx]
                
                # 1. Get Batch GT -> GPU
                b_gt_cpu = gt_mask[b_idx]
                b_gt = torch.from_numpy(b_gt_cpu).cuda(non_blocking=True).float()
                
                # 2. Get Batch Probs -> GPU
                batch_model_list = []
                for m in models_probs_list:
                    chunk_cpu = m[b_idx]
                    chunk_gpu = torch.from_numpy(chunk_cpu).cuda(non_blocking=True)
                    if chunk_cpu.dtype == np.uint8:
                        chunk_float = chunk_gpu.float() / 255.0
                    else:
                        chunk_float = chunk_gpu.float()
                    batch_model_list.append(chunk_float)
                
                # Stack (M, B, H, W)
                b_data = torch.stack(batch_model_list)
                
                # 3. Weighted Sum
                # (M, 1, 1, 1) * (M, B, H, W) -> Sum dim 0
                b_prob = (b_data * w_tensor).sum(dim=0)
                
                # 4. Metrics
                pred_mask = (b_prob > 0.5).float()
                
                total_inter += (pred_mask * b_gt).sum()
                total_union += pred_mask.sum() + b_gt.sum()
                
                del b_data, b_prob, pred_mask, b_gt
                
            dice = 2. * total_inter / (total_union + eps)
            return -dice.item()

    # Grid Search strategy
    if n_models <= 3: step = 0.05
    elif n_models == 4: step = 0.1
    else: step = 0.2
    
    best_score = 999
    best_weights = [1.0/n_models] * n_models
    
    # Grid Search Loop (Fast, no tqdm needed for inner)
    for w_cand in generate_simplex_grid(n_models, step):
        s = loss_func(w_cand)
        if s < best_score:
            best_score = s
            best_weights = w_cand
            
    # Nelder-Mead
    res = minimize(loss_func, best_weights, method='Nelder-Mead', bounds=[(0,1)]*n_models, tol=1e-3, options={'maxiter':20})
    
    final_w = res.x
    final_w = np.clip(final_w, 0, None)
    if final_w.sum() > 0:
        final_w /= final_w.sum()
        
    return final_w

def find_optimal_weights(models_probs, gt_masks):
    """
    Global Optimization Wrapper (Optimized)
    Uses N=50 subsampling to fit in RAM and speed up Grid Search.
    """
    n_models = len(models_probs)
    n_samples = gt_masks.shape[0]
    n_classes = gt_masks.shape[1]
    
    print(f">> [Robust] Optimizing Global Weights (Grid+NM)...")

    # --- OPTIMIZATION (GPU Accelerated + Batch-wise) ---
    # CPU calculation was too slow (Memory Bandwidth bound).
    # We use GPU for the heavy lifting (Cast -> Sum -> Dice).
    # Strategy: Keep Data on CPU (Uint8), Streaming Batches to GPU.
    
    indices = np.arange(n_samples)
    print(f"   -> Using ALL {n_samples} images for Global Optimization (GPU Accelerated)...")
    
    # Enable Cudnn Benchmark for speed if static size (which it is)
    torch.backends.cudnn.benchmark = True
    
    eps = 1e-6
    batch_size = 5 # Small batch size safe for GPU transfer
    
    # Define Vectorized Loss (GPU)
    def global_loss_func(weights):
        # weights: (M,) List or Array
        # Convert weights to GPU Tensor once
        w_tensor = torch.tensor(weights, device='cuda', dtype=torch.float32).view(-1, 1, 1, 1, 1) # (M, 1, 1, 1, 1)
        
        total_inter = torch.zeros(n_classes, device='cuda')
        total_union = torch.zeros(n_classes, device='cuda')
        
        # Iterate in batches
        with torch.no_grad(): # Disable Gradient tracking for speed
            for start_idx in range(0, n_samples, batch_size):
                 end_idx = min(start_idx + batch_size, n_samples)
                 b_idx = indices[start_idx:end_idx]
                 
                 # 1. Get Batch GT -> Move to GPU -> Float32
                 # gt_masks is numpy (N, C, H, W). 
                 # Copy overhead: 5*29*1024*1024 bytes = 150MB. Fast.
                 b_gt_cpu = gt_masks[b_idx]
                 b_gt = torch.from_numpy(b_gt_cpu).cuda(non_blocking=True).float()
                 
                 # 2. Get Batch Probs -> Move to GPU
                 # We need to construct a standard tensor for weighted sum
                 # models_probs is List of (N, C, H, W) numpy
                 
                 batch_model_list = []
                 for m in models_probs:
                     # Access CPU slice
                     chunk_cpu = m[b_idx]
                     # Move to GPU (Uint8)
                     chunk_gpu = torch.from_numpy(chunk_cpu).cuda(non_blocking=True)
                     # Cast if needed (on GPU is fast)
                     if chunk_cpu.dtype == np.uint8:
                         chunk_float = chunk_gpu.float() / 255.0
                     else:
                         chunk_float = chunk_gpu.float()
                     batch_model_list.append(chunk_float)
                 
                 # Stack: (M, B, C, H, W)
                 # This stack is on GPU VRAM. 2 Models * 150MB = 300MB. Safe.
                 b_data = torch.stack(batch_model_list)
                 
                 # 3. Weighted Sum
                 # (M, 1, 1, 1, 1) * (M, B, C, H, W) -> Sum Dim 0
                 b_prob = (b_data * w_tensor).sum(dim=0)
                 
                 # 4. Compute Metrics
                 pred_mask = (b_prob > 0.5).float()
                 
                 # Sum over (B, H, W) -> Result (C,)
                 # Axis mapping: (B, C, H, W) -> Sum dims (0, 2, 3)
                 total_inter += (pred_mask * b_gt).sum(dim=(0, 2, 3))
                 total_union += pred_mask.sum(dim=(0, 2, 3)) + b_gt.sum(dim=(0, 2, 3))
                 
                 # Cleanup VRAM explicitly (optional but good for loops)
                 del b_data, b_prob, pred_mask, b_gt, batch_model_list
            
            # Final Dice (Mean over classes)
            # total_inter is (C,) tensor
            dice_c = 2. * total_inter / (total_union + eps)
            score = dice_c.mean().item()
            
        return -score
        
        dice_c = 2. * intersection_c / (union_c + eps)
        return -dice_c.mean() # Mean over classes

    # Grid Search strategy same as above
    if n_models <= 3: step = 0.05
    elif n_models == 4: step = 0.1
    else: step = 0.2
    
    best_score = 999
    best_weights = [1.0/n_models] * n_models
    
    for w_cand in tqdm(generate_simplex_grid(n_models, step), desc="Grid Search"):
        s = global_loss_func(w_cand)
        if s < best_score:
            best_score = s
            best_weights = w_cand
            
    # Nelder-Mead
    res = minimize(global_loss_func, best_weights, method='Nelder-Mead', bounds=[(0,1)]*n_models, tol=1e-4, options={'maxiter':50})
    
    final_w = res.x
    final_w = np.clip(final_w, 0, None)
    if final_w.sum() > 0:
        final_w /= final_w.sum()

    print(f">> Best Global Dice: {-res.fun:.4f} | Weights: {final_w}")
    return final_w

def find_optimal_weights_per_class(models_probs, gt_masks):
    """
    Robust Class-wise Optimization
    """
    n_models = len(models_probs)
    n_classes = gt_masks.shape[1]
    
    print(f">> [Robust] Optimizing Weights PER CLASS (Grid+NM) for {n_models} models...")
    
    best_weights_matrix = np.zeros((n_classes, n_models))

    # Iterate over each class channel
    for c in tqdm(range(n_classes), desc="Optimizing Classes"):
        # Extract data for this class only
        # class_models_probs: List of (N, H, W)
        class_models_probs = [m[:, c, :, :] for m in models_probs] 
        class_gt = gt_masks[:, c, :, :]
        
        # Use simple robust finder for this single channel
        best_w = find_optimal_weights_robust(class_models_probs, class_gt)
        best_weights_matrix[c] = best_w
            
    print(">> Class-wise Optimization Done.")
    return best_weights_matrix

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

    # If Matrix, save JSON too
        if strategy == "classweighted":
             json_name = f"weights_matrix_{num_models}models.json"
             import json
             # Convert numpy to list
             w_list = weights.tolist()
             with open(json_name, "w") as f:
                 json.dump({"weights": w_list, "classes": Config.CLASSES}, f, indent=4)
             print(f"Saved Matrix Rules to: {json_name}")

class ConfigOverride:
    """
    Context Manager to temporarily override Config values 
    for a specific model's context.
    """
    def __init__(self, overrides):
        self.overrides = overrides
        self.backup = {}
        
    def __enter__(self):
        # 1. Store backup & Apply overrides
        for k, v in self.overrides.items():
            # Only override if keys exist in Config (Safety)
            # Case-insensitive matching could be implemented but strict is safer for now.
            # We map specific lowercase config keys from ENSEMBLE_MODELS to Config attributes.
            
            target_key = None
            if k == 'resize_size': target_key = 'RESIZE_SIZE'
            elif k == 'window_size': target_key = 'WINDOW_SIZE'
            elif k == 'stride': target_key = 'STRIDE'
            elif k == 'batch_size': target_key = 'BATCH_SIZE'
            elif k == 'scales': target_key = 'TTA_SCALES'
            elif k == 'tta_mode': target_key = 'TTA_MODE'
            # Add more mappings if needed
            
            if target_key and hasattr(Config, target_key):
                self.backup[target_key] = getattr(Config, target_key)
                setattr(Config, target_key, v)
                print(f"   [Config Override] {target_key} -> {v}")
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 2. Restore backup
        for k, v in self.backup.items():
            setattr(Config, k, v)
            
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
    # Config Check
    USE_OPTIMIZATION = getattr(Config, 'ENSEMBLE_USE_OPTIMIZATION', False)
    WEIGHT_METHOD = getattr(Config, 'ENSEMBLE_WEIGHT_METHOD', 'global') # 'global' or 'class'
    weights = getattr(Config, 'ENSEMBLE_WEIGHTS', None)

    # ====================================================
    # 1. Weight Optimization (Validation Step)
    # ====================================================
    
    if USE_OPTIMIZATION:
        print(f"\n[Phase 1] Auto-Finding Ensemble Weights ({WEIGHT_METHOD} mode)...")
        
        # 1. Load Reference Validation Data (Target GT)
        # Use Config.DATASET_FILE as the source of truth for GT
        # NOTE: GT is GT, it shouldn't be affected by model configs
        valid_loader, valid_filenames = get_validation_data(Config.DATASET_FILE)
        
        # OPTIMIZATION: Use 1024x1024 Resolution + UInt8 Storage
        # 2048 float32 = 435GB RAM (OOM)
        # 1024 uint8 = 27GB RAM (Safe)
        VALID_Optim_Size = 1024
        print(f"Loading GT Masks (Downsampling to {VALID_Optim_Size}x{VALID_Optim_Size})...")
        gt_dict = {} 
        
        current_idx = 0
        for batch in tqdm(valid_loader, desc="Loading GT Masks"):
            if isinstance(batch, (list, tuple)): masks = batch[1]
            elif isinstance(batch, dict): masks = batch['mask']
            
            if isinstance(masks, torch.Tensor):
                # GT should be float 0.0 or 1.0
                masks = F.interpolate(masks.float(), size=(VALID_Optim_Size, VALID_Optim_Size), mode='nearest')
                masks = masks.cpu().numpy().astype(np.uint8) # 0 or 1
            
            batch_size = masks.shape[0]
            for b in range(batch_size):
                if current_idx < len(valid_filenames):
                    fname = os.path.basename(valid_filenames[current_idx])
                    gt_dict[fname] = masks[b]
                    current_idx += 1
        
        # 2. Collect Predictions from All Models
        model_probs_dicts = [] 
        
        for i, cfg in enumerate(models_config):
            # === APPLY CONFIG OVERRIDE ===
            with ConfigOverride(cfg):
                # Now inside this block, Config is modified
                target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
                print(f">> [Valid] Model {i+1} | Dataset: {target_dataset_file}...")
                
                try:
                    # Reload Model each time to save GPU? Or keep loaded?
                    # Loading inside loop is safer.
                    model = load_model(cfg['path'])
                    inf_module_name = cfg['inference_file']
                    
                    curr_valid_loader, curr_valid_filenames = get_validation_data(target_dataset_file)
                    inference_adapter = ValidationWrapper(curr_valid_loader, curr_valid_filenames)
                    
                    try:
                        inf_module = importlib.import_module(inf_module_name)
                    except ModuleNotFoundError:
                        # Fallback: If 'inference.' prefix causes issue (running inside folder)
                        if inf_module_name.startswith("inference."):
                            inf_module = importlib.import_module(inf_module_name.replace("inference.", ""))
                        else:
                            raise
                    
                    # Filter kwargs for get_probs
                    exclude_keys = ['path', 'inference_file', 'dataset_file', 'scales', 'tta_mode', 'resize_size', 'batch_size']
                    inf_kwargs = {k: v for k, v in cfg.items() if k not in exclude_keys}
                    
                    # Return 1024 Probs
                    probs_dict = inf_module.get_probs(model, inference_adapter, return_downsampled=VALID_Optim_Size, **inf_kwargs)
                    
                    # Quantize to UInt8 to save RAM (0..255)
                    # probs_dict values are float 0..1
                    normalized_probs_dict = {}
                    for k, v in probs_dict.items():
                        # v might already be uint8 if optimized in get_probs
                        if v.dtype == np.uint8:
                            v_uint8 = v
                        else:
                            # v is (C, H, W) float
                            v_uint8 = (v * 255).astype(np.uint8)
                        
                        normalized_probs_dict[os.path.basename(k)] = v_uint8
                        
                    model_probs_dicts.append(normalized_probs_dict)
                    
                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error during optimization for Model {i+1}: {e}")
                    weights = None
                    model_probs_dicts = None
                    break
            # === EXIT OVERRIDE ===
            
        # 3. Intersection & Optimization
        if model_probs_dicts:
            common_keys = set(gt_dict.keys())
            for idx, d in enumerate(model_probs_dicts):
                common_keys &= set(d.keys())
            
            sorted_keys = sorted(list(common_keys))
            print(f">> Optimized Intersection: {len(sorted_keys)} files")
            
            if len(sorted_keys) > 0:
                 valid_gts = np.stack([gt_dict[k] for k in sorted_keys])
                 valid_model_probs = []
                 for d in model_probs_dicts:
                     valid_model_probs.append(np.stack([d[k] for k in sorted_keys]))
                 
                 # Optimization: Release huge dictionary memory
                 del model_probs_dicts
                 import gc; gc.collect()
                 
                 # BRANCH by Method
                 if WEIGHT_METHOD == 'class':
                     weights = find_optimal_weights_per_class(valid_model_probs, valid_gts)
                 else:
                     weights = find_optimal_weights(valid_model_probs, valid_gts)
        
        import gc; gc.collect()
        
    # --- Weight Handling Logic ---
    if weights is None:
        if Config.ENSEMBLE_WEIGHTS is not None:
             weights = Config.ENSEMBLE_WEIGHTS
        else:
             weights = [1.0/len(models_config)] * len(models_config)
        print(f"\n>> Final Ensemble Weights (Manual/Equal): {weights}")

    # Report & Save
    if isinstance(weights, np.ndarray) and weights.ndim == 2:
        print(f"\n>> Final Ensemble Weights (Class-wise Matrix): shape {weights.shape}")
        try:
            w_df = pd.DataFrame(weights, columns=[f"Model_{i+1}" for i in range(len(models_config))])
            w_df.insert(0, "Class", Config.CLASSES)
            w_path = f"ensemble_weights_{Config.EXPERIMENT_NAME}.csv"
            w_df.to_csv(w_path, index=False)
            print(f">> Saved learned weights to {w_path}")
        except Exception as e:
            print(f"Warning: Failed to save weights csv: {e}")
    else:
        print(f">> Final Weights: {weights}")
        
    # ====================================================
    # 2. Test Inference (Zero-Disk / In-Memory Mode)
    # ====================================================
    print("\n[Phase 2] Running Inference on Test Set (In-Memory Sequential Mode)...")
    
    # 1. Pre-load Models & Loaders
    loaded_models = []
    loaders = []
    modules = []
    
    print(">> Setting up models and loaders...")
    for i, cfg in enumerate(models_config):
        # Load Model to CPU to save VRAM
        print(f"   -> Loading Model {i+1} to CPU...")
        model = torch.load(cfg['path'], map_location='cpu', weights_only=False)
        model.eval()
        loaded_models.append(model)
        
        # Determine Module
        # Determine Module
        inf_module_name = cfg['inference_file']
        try:
            inf_module = importlib.import_module(inf_module_name)
        except ModuleNotFoundError:
            if inf_module_name.startswith("inference."):
                inf_module = importlib.import_module(inf_module_name.replace("inference.", ""))
            else:
                raise
        modules.append(inf_module)
        
        # Create Loader with Specific Config
        target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
        with ConfigOverride(cfg):
            ds_module = importlib.import_module(target_dataset_file)
            XRayInferenceDataset_cls = ds_module.XRayInferenceDataset
            get_transforms_fn = ds_module.get_transforms
            
            test_dataset = XRayInferenceDataset_cls(transforms=get_transforms_fn(is_train=False))
            # Batch size 1 is strictly required for this sequential logic
            # num_workers=0 is safer to avoid DALI/Fork issues and memory spikes
            loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            loaders.append(loader)
            
    print(f">> Ready to process {len(loaders[0])} images...")
    
    # 2. Sequential Inference Loop
    final_results = {}
    
    # Zip all loaders to iterate image by image
    # Note: Assumes all datasets have SAME order (sorted) - standard implementations do this.
    for batch_group in tqdm(zip(*loaders), total=len(loaders[0]), desc="Ensemble Inference"):
        # batch_group is a tuple of (images, image_names) from each loader
        
        # Check alignment (sanity check)
        # batch_group[0][1] is tuple of filenames for model 0
        ref_name = batch_group[0][1][0] 
        # Optional: Verify ref_name matches other models if paranoid
        
        base_filename = os.path.basename(ref_name)
        
        # Accumulator for this image
        final_prob_map = 0
        
        for i, (images, _) in enumerate(batch_group):
            # 1. Move Model to GPU
            model = loaded_models[i]
            model.cuda()
            
            # 2. Predict
            # images is (1, C, H, W)
            img_tensor = images[0] # Take first item of batch (since batch_size=1)
            
            # Predict Logic provided by module
            module = modules[i]
            cfg = models_config[i]
            
            pred_prob = None
            if hasattr(module, 'predict_one_image'):
                with torch.no_grad():
                    pred_prob = module.predict_one_image(model, img_tensor, **cfg)
                # Ensure it's (C, H, W) CPU Tensor (Wait, we want GPU)
                # predict_one_image in inference.py and sliding returns GPU tensor if logic is correct?
                # Actually predict_one_image returns Tensor. We check where it is.
                if isinstance(pred_prob, np.ndarray):
                    pred_prob = torch.from_numpy(pred_prob).cuda()
                elif not pred_prob.is_cuda:
                    pred_prob = pred_prob.cuda()
            else:
                print(f"Error: {module} missing 'predict_one_image'")
                model.cpu()
                continue
            
            # 4. Resize & Accumulate (GPU Optimized)
            if pred_prob.shape[1] != 2048 or pred_prob.shape[2] != 2048:
                pred_prob = F.interpolate(pred_prob.unsqueeze(0), size=(2048, 2048), mode='bilinear', align_corners=False).squeeze(0)
                
            # Apply Weights
            w_val = 0
            if isinstance(weights, np.ndarray) and weights.ndim == 2:
                w_vec = weights[:, i] 
                w_tensor = torch.tensor(w_vec, device='cuda', dtype=torch.float32).view(-1, 1, 1)
                weighted_prob = pred_prob * w_tensor
            else:
                w_val = weights[i]
                weighted_prob = pred_prob * w_val
                
            # Accumulate to GPU Buffer
            if isinstance(final_prob_map, int) and final_prob_map == 0:
                final_prob_map = weighted_prob
            else:
                final_prob_map += weighted_prob
            
            # Offload Model
            model.cpu()
            del pred_prob, weighted_prob
            torch.cuda.empty_cache()
                    
        # 5. Threshold & Encode
        # Now move final result to CPU
        pred_mask = (final_prob_map > 0.5) # Boolean Tensor on GPU
        
        # Move to CPU for RLE
        for c, segm in enumerate(pred_mask):
            rle = encode_mask_to_rle(segm.cpu().numpy())
            class_name = Config.CLASSES[c]
            final_results[f"{class_name}_{base_filename}"] = rle
            
    # 3. Save CSV
    print(">> Saving Final Submission...")
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(final_results.get(key, ""))
        sample_df['rle'] = final_rles
        
        # Build Filename
        num_models = len(models_config)
        
        w_str = ""
        if isinstance(weights, list) or (isinstance(weights, np.ndarray) and weights.ndim==1):
            w_str = "-".join([f"{w:.2f}" for w in weights])
        else:
             w_str = "matrix"
             
        save_name = f"submission_ens_{num_models}models_{WEIGHT_METHOD}_{w_str}.csv"
        sample_df.to_csv(save_name, index=False)
        print(f"Done! Saved to {save_name}")

if __name__ == '__main__':
    test()
