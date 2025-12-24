import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import os
import importlib
import numpy as np
import torchvision.transforms.functional as TF

from config import Config
from utils import encode_mask_to_rle

# ============================================================================
# TTA Functions
# ============================================================================
def tta_scale_only(model, images, device='cuda'):
    """
    Multi-Scale TTA only
    
    Time: 3x
    Expected improvement: +0.001~0.002
    """
    model.eval()
    batch_size = images.shape[0]
    original_size = images.shape[2:]  # (H, W)
    
    all_predictions = []
    
    with torch.no_grad():
        for scale in Config.TTA_SCALES:
            if scale != 1.0:
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)
                scaled_images = F.interpolate(
                    images,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_images = images
            
            # Inference
            with torch.amp.autocast(device_type="cuda"):
                preds = model(scaled_images)
                if isinstance(preds, dict):
                    preds = preds['out']
            
            # Resize back to original
            if scale != 1.0:
                preds = F.interpolate(
                    preds,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            all_predictions.append(preds)
        
        # Average predictions
        averaged_pred = torch.stack(all_predictions).mean(dim=0)
    
    return averaged_pred

def tta_small_angle(model, images, device='cuda'):
    """
    Small Angle TTA only
    
    Time: 3x
    Expected improvement: +0.002~0.003
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for angle in Config.TTA_ANGLES:
            if angle == 0:
                rotated = images
            else:
                # Rotate (for each image)
                rotated = torch.stack([
                    TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                    for img in images
                ])
            
            # Inference
            with torch.amp.autocast(device_type="cuda"):
                preds = model(rotated)
                if isinstance(preds, dict):
                    preds = preds['out']
            
            # Rotate back
            if angle != 0:
                preds = torch.stack([
                    TF.rotate(pred, -angle, interpolation=TF.InterpolationMode.BILINEAR)
                    for pred in preds
                ])
            
            all_predictions.append(preds)
        
        # Average predictions
        averaged_pred = torch.stack(all_predictions).mean(dim=0)
    
    return averaged_pred

def tta_balanced(model, images, device='cuda'):
    """
    Multi-Scale + Small Angle TTA
    
    Time: 9x (3 scales × 3 angles)
    Expected improvement: +0.003~0.005
    """
    model.eval()
    original_size = images.shape[2:]
    all_predictions = []
    
    with torch.no_grad():
        for scale in Config.TTA_SCALES:
            # Scale
            if scale != 1.0:
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)
                scaled_images = F.interpolate(
                    images,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_images = images
            
            for angle in Config.TTA_ANGLES:
                # Rotate
                if angle == 0:
                    rotated = scaled_images
                else:
                    rotated = torch.stack([
                        TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                        for img in scaled_images
                    ])
                
                # Inference
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(rotated)
                    if isinstance(preds, dict):
                        preds = preds['out']
                
                # Rotate back
                if angle != 0:
                    preds = torch.stack([
                        TF.rotate(pred, -angle, interpolation=TF.InterpolationMode.BILINEAR)
                        for pred in preds
                    ])
                
                # Resize back
                if scale != 1.0:
                    preds = F.interpolate(
                        preds,
                        size=original_size,
                        mode='bilinear',
                        align_corners=False
                    )
                
                all_predictions.append(preds)
        
        # Average predictions
        averaged_pred = torch.stack(all_predictions).mean(dim=0)
    
    return averaged_pred

def simple_inference(model, images):
    """Simple Inference (No TTA)"""
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
    return outputs

def tta_inference(model, images, device='cuda'):
    """
    Call appropriate function based on TTA strategy
    """
    if Config.TTA_TYPE == 'scale_only':
        return tta_scale_only(model, images, device)
    elif Config.TTA_TYPE == 'small_angle':
        return tta_small_angle(model, images, device)
    elif Config.TTA_TYPE == 'balanced':
        return tta_balanced(model, images, device)
    else:
        print(f"Unknown TTA_TYPE: {Config.TTA_TYPE}, using simple inference")
        return simple_inference(model, images)

# ============================================================================
# Main Test Function
# ============================================================================
def test():
    # 1. Load modules and model
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError:
        print("Error loading modules. Check config.py")
        return None
        
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    print(f"Loading Model from {model_path}")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()

    # 2. Data loader
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # Dictionary to store results
    results_dict = {}
    
    # 3. Print TTA settings
    if Config.USE_TTA:
        print(f"===== TTA Enabled =====")
        print(f"TTA Type: {Config.TTA_TYPE}")
        if Config.TTA_TYPE in ['scale_only', 'balanced']:
            print(f"Scales: {Config.TTA_SCALES}")
        if Config.TTA_TYPE in ['small_angle', 'balanced']:
            print(f"Angles: {Config.TTA_ANGLES}")
        print(f"=======================")
    else:
        print("===== Simple Inference (No TTA) =====")
    
    # 4. Start inference
    print("Start Inference...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            images = images.cuda()
            
            # ===== TTA branching =====
            if Config.USE_TTA:
                outputs = tta_inference(model, images, device='cuda')
            else:
                outputs = simple_inference(model, images)
            # ====================
            
            # Restore to original size (2048, 2048)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).detach().cpu().numpy()
            
            for output, image_path in zip(outputs, image_names):
                # Extract only the filename without path
                image_name = os.path.basename(image_path)
                
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    class_name = Config.CLASSES[c]
                    results_dict[f"{class_name}_{image_name}"] = rle

    # 5. Match with sample_submission.csv
    print("Post-processing and Matching with sample_submission.csv...")
    sample_sub_path = "sample_submission.csv"
    
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        
        # Fill RLE values according to sample_submission order
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            final_rles.append(results_dict.get(key, ""))
            
        sample_df['rle'] = final_rles
        final_df = sample_df
    else:
        print("Warning: sample_submission.csv not found!")
        final_df = pd.DataFrame([
            {"image_name": k.split('_', 1)[1], "class": k.split('_', 1)[0], "rle": v}
            for k, v in results_dict.items()
        ])

    # 6. Final save
    tta_suffix = f"_TTA_{Config.TTA_TYPE}" if Config.USE_TTA else ""
    save_path = f"submission_{Config.EXPERIMENT_NAME}{tta_suffix}.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    
    return save_path  # Return the saved filename

if __name__ == '__main__':
    saved_file = test()
    if saved_file:
        print(f"\n>>> [Stage 2] Inference Completed Successfully.")
        print(f">>> All processes finished. Check {saved_file}")
    else:
        print("\n>>> [Stage 2] Inference Failed.")


