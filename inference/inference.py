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

def predict_one_image(model, image, **kwargs):
    """
    Predict on a single image tensor (C, H, W).
    Returns: fused_prob (C, H, W) on GPU/CPU (Tensor)
    """
    # image: (C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.cuda()
    
    # Use config from kwargs if needed? Standard inference usually just forward pass
    # model is already in eval mode
    
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(image)
        if isinstance(outputs, dict): outputs = outputs['out']
    
    # Resize to Canoncial Size (2048)
    outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
    
    # Sigmoid -> Probs
    prob = torch.sigmoid(outputs)
    
    return prob.squeeze(0)


def get_probs(model, loader, save_dir=None, return_downsampled=None, save_dtype='float16', **kwargs):
    """
    Standard inference capable of:
    1. Saving to Disk (for Ensemble Phase 2)
    2. Returning Downsampled Arrays (for Ensemble Phase 1)
    3. Returning RLE Strings (Default Standalone)
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    print(f">> [inference] Standard Inference")
    if save_dir: print(f"   -> Saving results to {save_dir} (float16)")
    if return_downsampled: print(f"   -> Returning downsampled results (Size: {return_downsampled})")
    
    results = {}
    model.eval()
    
    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Inference"):
            images = images.cuda()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
            
            # Resize to Canoncial Size (2048)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs)
            
            for output, image_path in zip(outputs, image_names):
                fname = os.path.basename(image_path)
                
                # 1. Downsampled Return
                if return_downsampled:
                    # Resize
                    small_prob = F.interpolate(output.unsqueeze(0), size=(return_downsampled, return_downsampled), mode="bilinear", align_corners=False).squeeze()
                    # Optimize Memory: unit8
                    results[fname] = (small_prob.cpu().numpy() * 255).astype(np.uint8)
                    
                # 2. Disk Save (Float16)
                elif save_dir:
                    if save_dtype == 'uint8':
                        # Optimize Memory: uint8 (0-255)
                        prob_np = (output.cpu().numpy() * 255).astype(np.uint8)
                        np.save(os.path.join(save_dir, fname + ".npy"), prob_np)
                    else:
                        prob_np = output.half().cpu().numpy()
                        np.save(os.path.join(save_dir, fname + ".npy"), prob_np)
                    
                # 3. Default: RLE Encoding (RAM Safe)
                else:
                    pred_mask = (output > 0.5)
                    for c, segm in enumerate(pred_mask):
                        rle = encode_mask_to_rle(segm.cpu().numpy())
                        class_name = Config.CLASSES[c]
                        results[f"{class_name}_{fname}"] = rle
                
    return results

def predict_one_image(model, image, **kwargs):
    # Ensure image is on CUDA
    image = image.cuda()
    if image.dim() == 3: image = image.unsqueeze(0)
    
    with torch.amp.autocast(device_type="cuda"):
        output = model(image)
        if isinstance(output, dict): output = output['out']
    
    output = F.interpolate(output, size=(2048, 2048), mode="bilinear", align_corners=False)
    return torch.sigmoid(output).squeeze(0) # Return Probs

def test():
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError: 
        return
    
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    # ============================================================
    # [MODIFIED] Fine-tuning 지원: 모델 경로 자동 선택
    # ============================================================
    if Config.USE_FINETUNE:
        model_filename = "finetuned_model.pt"
        model_type = "Fine-tuned"
    else:
        model_filename = "best_model.pt"
        model_type = "Best"
    
    model_path = os.path.join(Config.SAVED_DIR, model_filename)
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ {model_type} model not found: {model_path}")
        if Config.USE_FINETUNE:
            print("   Please run training with USE_FINETUNE=True first to create finetuned_model.pt")
        else:
            print("   Please run training with USE_FINETUNE=False first to create best_model.pt")
        return
    
    # 모델 로딩
    print(f">> Loading {model_type} Model from: {model_path}")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    print(f"✅ {model_type} model loaded successfully!")
    
    # Dataset & Loader
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # Inference
    results_dict = get_probs(model, test_loader)
    
    # ============================================================
    # [MODIFIED] CSV 저장 - Fine-tuning 구분
    # ============================================================
    sample_sub_path = "sample_submission.csv"
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        final_rles = []
        for _, row in sample_df.iterrows():
            key = f"{row['class']}_{row['image_name']}"
            rle = results_dict.get(key, "")
            final_rles.append(rle)
        
        # Fine-tuning 모드에 따라 파일명 구분
        if Config.USE_FINETUNE:
            save_path = f"submission_{Config.EXPERIMENT_NAME}_finetune.csv"
        else:
            save_path = f"submission_{Config.EXPERIMENT_NAME}.csv"
        
        sample_df['rle'] = final_rles
        sample_df.to_csv(save_path, index=False)
        print(f"✅ Saved: {save_path}")
    else:
        print("❌ sample_submission.csv not found.")

if __name__ == '__main__':
    test()