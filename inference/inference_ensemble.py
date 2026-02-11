
import sys
import os

# [Fix] Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import importlib
import warnings
import json
import gc
import cv2
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
from scipy.optimize import minimize
from config import Config
from itertools import islice
from utils import encode_mask_to_rle

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Helper: ConfigOverride
# =============================================================================
class ConfigOverride:
    def __init__(self, overrides):
        self.overrides = overrides
        self.backup = {}
        
    def __enter__(self):
        for k, v in self.overrides.items():
            target_key = None
            if k == 'resize_size': target_key = 'RESIZE_SIZE'
            elif k == 'window_size': target_key = 'WINDOW_SIZE'
            elif k == 'stride': target_key = 'STRIDE'
            elif k == 'batch_size': target_key = 'BATCH_SIZE'
            elif k == 'scales': target_key = 'TTA_SCALES'
            elif k == 'tta_mode': target_key = 'TTA_MODE'
            elif k == 'NUM_WORKERS' or k == 'num_workers': target_key = 'NUM_WORKERS'
            
            if target_key and hasattr(Config, target_key):
                self.backup[target_key] = getattr(Config, target_key)
                setattr(Config, target_key, v)
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.backup.items():
            setattr(Config, k, v)

# =============================================================================
# Helper: Full Image Loader (For Sliding Window Reconstruction)
# =============================================================================
def get_full_image_data():
    """
    Load validation data as Full Images (not patches).
    """
    try:
        mod = importlib.import_module("dataset.dataset_dali_exclude")
    except ModuleNotFoundError:
        print("!! [Error] dataset.dataset_dali_exclude not found. Required for Full Image loading.")
        raise
        
    loader_wrapper = mod.get_dali_loader(is_train=False, batch_size=1)
    filenames = loader_wrapper.source.filenames
    return loader_wrapper, filenames

# =============================================================================
# Core Pipeline (Generic Interface)
# =============================================================================
class EnsemblePipeline:
    def __init__(self):
        self.models_config = self._parse_config()
        self.num_classes = len(Config.CLASSES)
        self.classes = Config.CLASSES
        
        # Optimization Params
        self.use_optim = getattr(Config, 'ENSEMBLE_USE_OPTIMIZATION', False)
        self.weight_method = getattr(Config, 'ENSEMBLE_WEIGHT_METHOD', 'class')
        
        # Storage for Final Weights
        num_models = len(self.models_config)
        
        # Initial Weights (from Config or Equal)
        if hasattr(Config, 'ENSEMBLE_WEIGHTS') and Config.ENSEMBLE_WEIGHTS is not None:
             # Broadcast to matrix (C, M)
             w_init = np.array(Config.ENSEMBLE_WEIGHTS)
             if w_init.ndim == 1:
                 # (M,) -> (C, M)
                 self.final_weights = np.tile(w_init, (self.num_classes, 1))
             else:
                 self.final_weights = w_init
        else:
             self.final_weights = np.ones((self.num_classes, num_models)) / num_models
        
    def _parse_config(self):
        models = getattr(Config, 'ENSEMBLE_MODELS', [])
        print(f">> Found {len(models)} models in Config.")
        return models
        
    def load_model(self, path):
        import torch
        # [Fix] PyTorch 2.6 requires weights_only=False for full model pickle
        model = torch.load(path, weights_only=False)
        if hasattr(model, 'module'):
            model = model.module
        return model # Retain on CPU initially to prevent OOM
        # We will move to CUDA on demand

    def run_phase1_optimization(self):
        if not self.use_optim:
            print(">> Optimization Disabled. Skipping Phase 1 (Using Config/Equal Weights).")
            return

        print(f"\n[Phase 1] Starting Generic Ensemble Optimization (V2 - Local High Res)")
        print(f"   -> Method: Optimize on Fully Reconstructed Probability Maps (2048x2048).")
        
        # 1. Setup Subsampling
        MAX_IMAGES = 50 
        print(f"   -> [Constraint] Subsampling to {MAX_IMAGES} images (Randomized).")
        
        # 2. Load GT (Reconstructed / Full)
        # [Fix] Enforce User's Optimization Resolution (e.g. 2048)
        optim_size = getattr(Config, 'ENSEMBLE_OPTIM_SIZE', 2048)
        original_resize = Config.RESIZE_SIZE
        Config.RESIZE_SIZE = (optim_size, optim_size)
        print(f"   -> [Config] Forcing Validation Resolution to {Config.RESIZE_SIZE}")
        
        try:
             loader, filenames = get_full_image_data()
        except:
             return
             
        # [Randomization Logic]
        total_valid_images = len(filenames)
        if total_valid_images > MAX_IMAGES:
             selected_indices = set(random.sample(range(total_valid_images), MAX_IMAGES))
             print(f"   -> [Random] Selected {MAX_IMAGES} indices from {total_valid_images} validation images.")
        else:
             selected_indices = set(range(total_valid_images))
             print(f"   -> [Info] Validation set size ({total_valid_images}) <= MAX_IMAGES. Using all.")

        gt_maps = {} 
        # Cache Inputs not needed for Phase 1 as we load maps via get_probs?
        # Actually in our logic, we load maps by running inference on these images.
        # We need to run inference on SPECIFIC 50 images.
        # So we better cache the INPUT IMAGES (Tensor) to feed to models.
        
        images_cache = [] 
        selected_fnames = []
        
        print(f"   -> Loading GT Maps & Caching Images (Unified)...")
        idx = 0
        for images, masks in tqdm(loader, total=len(loader), desc="Loading Data"):
            if idx not in selected_indices:
                idx += 1
                continue
                
            fname = os.path.basename(filenames[idx])
            mask_np = masks[0].cpu().numpy().astype(np.uint8)
            gt_maps[fname] = mask_np
            
            img_cpu = images[0].cpu() # Tensor
            images_cache.append(img_cpu)
            selected_fnames.append(fname)
            
            idx += 1
            
        print(f"   -> Loaded {len(selected_fnames)} GT Maps and Cached {len(images_cache)} Images.")

        # 3. Get Model Probabilities (Chunked by Model)
        # Structure: model_maps[model_idx] = {fname: prob_map (C, H, W)}
        model_maps = [{} for _ in range(len(self.models_config))]
        
        # Create a Loader for Cached Images
        # Simple Iterator yielding (image, fname)
        class ChunkDataset(torch.utils.data.Dataset):
            def __init__(self, imgs, fnames):
                self.imgs = imgs
                self.fnames = fnames
            def __len__(self): return len(self.imgs)
            def __getitem__(self, i): return self.imgs[i], self.fnames[i]
            
        chunk_dataset = ChunkDataset(images_cache, selected_fnames)
        # Batch size 4 for inference
        chunk_loader = torch.utils.data.DataLoader(chunk_dataset, batch_size=4, shuffle=False, num_workers=4)

        for m_idx, model_cfg in enumerate(self.models_config):
            print(f"   -> [Model {m_idx+1}] Generating Maps...")
            
            with ConfigOverride(model_cfg):
                model = self.load_model(model_cfg['path']).cuda()
                model.eval()
                
                inf_module_name = model_cfg.get('inference_file', 'inference.inference_sliding')
                try:
                    # Import with Fallback
                    try:
                        inf_module = importlib.import_module(inf_module_name)
                    except ModuleNotFoundError:
                        if "inference." in inf_module_name:
                            inf_module = importlib.import_module(inf_module_name.split("inference.")[-1])
                        else:
                            raise
                    
                    # Inference Logic
                    exclude_keys = ['path', 'inference_file', 'dataset_file', 'scales', 'tta_mode', 'resize_size']
                    inf_kwargs = {k: v for k, v in model_cfg.items() if k not in exclude_keys}
                    
                    probs_dict = inf_module.get_probs(model, chunk_loader, return_downsampled=Config.RESIZE_SIZE[0], **inf_kwargs)
                    model_maps[m_idx] = probs_dict
                    
                except Exception as e:
                    print(f"Error inference model {m_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
                del model
                torch.cuda.empty_cache()
                gc.collect()

        # 4. Optimize
        print(f"   -> [Optimization] Running V2 Global Optimization...")
        self._optimize_reconstructed_v2(gt_maps, model_maps, selected_fnames)
        
        # Restore Config
        Config.RESIZE_SIZE = original_resize
        
    def _optimize_reconstructed_v2(self, gt_maps, model_maps, fnames):
        """
        V2 Optimization:
        1. Calculates Intersection, Union, GT_Sum stats for all classes.
        2. Solves Optimization Problem (Maximize Mean Dice).
        """
        num_models = len(model_maps)
        
        # Stats Storage
        # I_mn[m, c] -> Intersection for model m, class c (over all images)
        # U_mn[m, c] -> Union Area for model m prediction (sum of probs)
        # G_n[c] -> GT Sum Area
        # Actually we need stats PER IMAGE to properly sum them?
        # Dice = 2 * (w . Inter) / (w . Union + G)
        # Sum is over pixels. We can sum over images too (Micro Dice) or per image (Macro)?
        # Competition Metric is Mean Dice per Image? Or Per Dataset?
        # Standard Dice Loss is Per Image normally.
        # Evaluation metric: "Mean Dice Coefficient". Usually average over all test images.
        # So we should Optimize: Mean(Dice(Image_i)).
        
        # To do this efficienty, we need I_mic, U_mic (Model m, Image i, Class c).
        
        print(f"      -> [V2] Collecting Stats (Intersection/Union) for {len(fnames)} images...")
        
        # Storage: List of Dicts? Or huge array?
        # M models, N images, C classes.
        # 3 * 50 * 29 = 4350. Small.
        M = num_models
        N = len(fnames)
        C = self.num_classes
        
        stats_I = np.zeros((N, C, M), dtype=np.float32) # Intersection
        stats_U = np.zeros((N, C, M), dtype=np.float32) # Pred Sum
        stats_G = np.zeros((N, C), dtype=np.float32)    # GT Sum
        
        for i, fname in enumerate(tqdm(fnames, desc="Stats")):
            gt = gt_maps[fname] # (C, H, W) uint8
            
            # Pre-calculate G
            # gt is 0 or 1
            # Sum per class
            g_sum = gt.sum(axis=(1, 2))
            stats_G[i] = g_sum
            
            for m in range(M):
                if fname not in model_maps[m]: continue
                
                prob = model_maps[m][fname] # (C, H, W) float or uint8
                if prob.dtype == np.uint8:
                    prob = prob.astype(np.float32) / 255.0
                
                # Inter = Prob * GT
                # Union = Prob
                
                # Vectorized Over Classes
                # prob * gt
                inter = (prob * gt).sum(axis=(1, 2))
                union = prob.sum(axis=(1, 2))
                
                stats_I[i, :, m] = inter
                stats_U[i, :, m] = union
                
        # Define Solver for a Set of Stats (Optimization Target)
        def solve(target_I, target_U, target_G):
            # target_I: (N, M) - Intersection for specific class(es) per image
            # but wait. If Global, we avg over classes too.
            # Loss = - Mean_over_images( Mean_over_classes( Dice(w) ) )
            
            # Let's vectorize efficiently. 
            # w: (M,)
            # Dice_ic(w) = 2 * (w . I_ic) / (w . U_ic + G_ic)
            # Score(w) = Mean_i( Mean_c ( Dice_ic(w) ) )
            
            # 1. Grid Search
            import itertools
            step = 0.2
            steps = np.arange(0, 1.0 + 1e-5, step)
            
            best_grid_score = -1
            best_grid_w = np.ones(M) / M
            
            # Pre-filter steps to sum=1
            valid_combs = [c for c in itertools.product(steps, repeat=M) if abs(sum(c)-1.0)<1e-5]
            
            for w in valid_combs:
                w_arr = np.array(w)
                # Compute Score
                # w . I -> (N, C)
                # w . U -> (N, C)
                
                # Weighted I, U
                w_I = np.dot(target_I, w_arr) # (N, C, M) . (M) -> (N, C)
                w_U = np.dot(target_U, w_arr)
                
                dice = 2 * w_I / (w_U + target_G + 1e-6)
                score = dice.mean()
                
                if score > best_grid_score:
                    best_grid_score = score
                    best_grid_w = w_arr

            # 2. Nelder-Mead
            def loss(w):
                # Softmax or Clip?
                # Using simple clipping normalization
                w = np.array(w)
                w = np.clip(w, 0, None)
                if w.sum() == 0: return 1.0
                w /= w.sum()
                
                w_I = np.dot(target_I, w)
                w_U = np.dot(target_U, w)
                dice = 2 * w_I / (w_U + target_G + 1e-6)
                return -dice.mean()
                
            init_w = best_grid_w
            res = minimize(loss, init_w, method='Nelder-Mead', tol=1e-4)
            
            final_w = res.x
            final_w = np.clip(final_w, 0, None)
            if final_w.sum() > 0: final_w /= final_w.sum()
            
            return final_w, -res.fun

        # Run Optimization
        if self.weight_method == 'global':
            print(f"      -> Running Global Optimization (All Classes jointly)...")
            # Pass Full Stats (N, C, M)
            best_w, score = solve(stats_I, stats_U, stats_G)
            print(f"      -> Global Best: {best_w}, Mean Dice: {score:.4f}")
            self.final_weights[:] = best_w # Broadcast (3,) -> (29, 3)
            
        else: # Class-wise
            print(f"      -> Running Class-wise Optimization...")
            for c in range(C):
                # Slice (N, M)
                c_I = stats_I[:, c, :]
                c_U = stats_U[:, c, :]
                c_G = stats_G[:, c]
                # We need to reshape c_G to (N, 1) to match solve logic? 
                # Solve logic: w_I = dot(I, w). I is (N, C, M).
                # If we pass (N, M) as I. Then w_I is (N,).
                # c_G is (N,).
                # Matches.
                
                # However, solve expects dimensions to match.
                # Let's adjust solve to handle (N, M) input nicely or enforce (N, 1, M).
                # My solve implementation uses `target_I` dot `w`.
                # If target_I is (N, C, M), result is (N, C).
                # If target_I is (N, M), result is (N,).
                # Dice calc is elementwise. .mean() works.
                # So it works for both.
                
                best_w, score = solve(c_I, c_U, c_G)
                self.final_weights[c] = best_w
                # print(f"Class {c}: {best_w}")

    def run_phase2_inference(self):
        print(f"\n[Phase 2] Running Inference on Test Set (In-Memory Sequential Mode - Github Style)...")
        
        # 1. Setup Test Dataset & Models (CPU)
        loaded_models = []
        loaders = []
        modules = []
        
        print(">> Setting up models and loaders...")
        for i, cfg in enumerate(self.models_config):
            # Load Model to CPU (RAM)
            print(f"   -> Loading Model {i+1} to CPU...")
            model = self.load_model(cfg['path'])
            model.eval()
            loaded_models.append(model)
            
            # Module
            inf_module_name = cfg.get('inference_file', 'inference.inference_tta')
            try:
                inf_module = importlib.import_module(inf_module_name)
            except:
                # Retry
                if inf_module_name.startswith("inference."):
                    inf_module = importlib.import_module(inf_module_name.replace("inference.", ""))
                else: raise

            # Check for predict_one_image
            if not hasattr(inf_module, 'predict_one_image'):
                 print(f"!! Error: Module {inf_module_name} missing 'predict_one_image'. Cannot run In-Memory Mode.")
                 return
            modules.append(inf_module)
            
            # Dataset (Standard Inference Dataset)
            # Use ConfigOverride to set Transforms/etc
            with ConfigOverride(cfg):
                # Import Dataset
                target_dataset_file = cfg.get('dataset_file', Config.DATASET_FILE)
                ds_mod = importlib.import_module(target_dataset_file)
                # Standard Inference Dataset
                if hasattr(ds_mod, 'XRayInferenceDataset'):
                    DatasetCls = ds_mod.XRayInferenceDataset
                    get_transforms = ds_mod.get_transforms
                    test_dataset = DatasetCls(transforms=get_transforms(is_train=False))
                    # Batch Size 1 Strict
                    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
                    loaders.append(loader)
                else:
                    print(f"!! Error: Dataset {target_dataset_file} missing XRayInferenceDataset.")
                    return

        print(f">> Ready to process {len(loaders[0])} images...")
        
        final_results = {}
        
        # 2. Sequential Inference Loop
        # Zip loaders
        count = 0
        for batch_group in tqdm(zip(*loaders), total=len(loaders[0]), desc="Ensemble Inference"):
            # batch_group: tuple of (images, filenames)
            
            # Base Filename
            ref_names = batch_group[0][1]
            if isinstance(ref_names, tuple): ref_names = ref_names[0]
            base_filename = os.path.basename(ref_names)
            
            final_prob_map = None
            
            # Loop Models
            for i, (images, _) in enumerate(batch_group):
                # images: (1, 3, H, W)
                img_tensor = images[0]
                
                # Move Model to GPU
                model = loaded_models[i]
                model.cuda()
                
                # Predict
                module = modules[i]
                cfg = self.models_config[i]
                
                # Filter kwargs needed? 
                # predict_one_image usually takes (model, image, **kwargs)
                # We pass entire cfg
                with torch.no_grad():
                    # Result: (C, H, W) Tensor PROBS
                    pred_prob = module.predict_one_image(model, img_tensor, **cfg)
                
                # Ensure on GPU and correct size
                if not pred_prob.is_cuda: pred_prob = pred_prob.cuda()
                
                if pred_prob.shape[1] != 2048 or pred_prob.shape[2] != 2048:
                     pred_prob = F.interpolate(pred_prob.unsqueeze(0), size=(2048, 2048), mode='bilinear', align_corners=False).squeeze(0)
                
                # Apply Weights
                # self.final_weights: (C, M)
                # i-th model weights: (C,)
                w_vec = self.final_weights[:, i]
                w_tensor = torch.from_numpy(w_vec).float().cuda().view(-1, 1, 1) # (C, 1, 1)
                
                weighted = pred_prob * w_tensor
                
                if final_prob_map is None:
                    final_prob_map = weighted
                else:
                    final_prob_map += weighted
                    
                # Offload
                model.cpu()
                del pred_prob, weighted, w_tensor
                # torch.cuda.empty_cache() # Optional, might slow down if too frequent
                
            # Post-Process (Threshold & RLE)
            # final_prob_map: (C, H, W)
            pred_mask = (final_prob_map > 0.5) # Bool
            
            # CPU RLE
            pred_mask_np = pred_mask.cpu().numpy()
            
            for c, segm in enumerate(pred_mask_np):
                rle = encode_mask_to_rle(segm)
                class_name = self.classes[c]
                final_results[f"{class_name}_{base_filename}"] = rle
            
            count += 1
            if count % 50 == 0:
                gc.collect()

        # 3. Save CSV
        print(">> Saving Final Submission...")
        
        # Load Sample Submission to maintain order
        sample_sub_path = "./sample_submission.csv"
        # Search for it? Assuming standard location
        if not os.path.exists(sample_sub_path):
             # Try parent
             sample_sub_path = "../sample_submission.csv"
        
        if os.path.exists(sample_sub_path):
            sample_df = pd.read_csv(sample_sub_path)
            final_rles = []
            for _, row in sample_df.iterrows():
                key = f"{row['class']}_{row['image_name']}"
                if key in final_results:
                    final_rles.append(final_results[key])
                else:
                    final_rles.append("")
            sample_df['rle'] = final_rles
            
            save_name = f"outputs/{Config.EXPERIMENT_NAME}_ensemble_v2.csv"
            if not os.path.exists("outputs"): os.makedirs("outputs")
            
            sample_df.to_csv(save_name, index=False)
            print(f"Done! Saved to {save_name}")
        else:
            print("Warning: sample_submission.csv not found. Saved dict as json dump.")
            with open("outputs/ensemble_results_raw.json", "w") as f:
                json.dump(final_results, f)

if __name__ == '__main__':
    pipeline = EnsemblePipeline()
    pipeline.run_phase1_optimization()
    pipeline.run_phase2_inference()
