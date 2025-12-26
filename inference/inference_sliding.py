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

def test():
    # 1. 모듈 및 모델 로드
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
    except ModuleNotFoundError:
        print("Error loading modules. Check config.py")
        return
        
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    
    model_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading Model from {model_path}")
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()

    # 2. 데이터 로더 (배치 사이즈 1 - 이미지당 처리)
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Sliding Window 설정
    window_size = getattr(Config, 'WINDOW_SIZE', 1024)
    stride = getattr(Config, 'STRIDE', 1024)
    
    print(f"[Sliding Window] Window: {window_size}x{window_size}, Stride: {stride}")
    
    # 결과를 저장할 딕셔너리
    results_dict = {}
    
    # 3. 인퍼런스 시작
    print("Start Inference with Sliding Window...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            # images: (1, 3, 2048, 2048)
            image = images[0]  # (3, 2048, 2048)
            image_name = image_names[0]
            
            # 슬라이딩 윈도우로 패치 추출
            patches = []
            positions = []
            
            for y in range(0, 2048 - window_size + 1, stride):
                for x in range(0, 2048 - window_size + 1, stride):
                    patch = image[:, y:y+window_size, x:x+window_size]  # (3, 1024, 1024)
                    patches.append(patch)
                    positions.append((y, x))
            
            # 패치를 배치로 묶기
            patches = torch.stack(patches).cuda()  # (N, 3, 1024, 1024)
            
            # 모델 예측
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(patches)  # (N, 29, 1024, 1024)
                if isinstance(outputs, dict): 
                    outputs = outputs['out']
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).cpu().numpy()  # (N, 29, 1024, 1024)
            
            # 패치 재조립 (2048x2048)
            full_mask = np.zeros((len(Config.CLASSES), 2048, 2048), dtype=np.uint8)
            
            for (y, x), patch_output in zip(positions, outputs):
                # patch_output: (29, 1024, 1024)
                full_mask[:, y:y+window_size, x:x+window_size] = patch_output
            
            # RLE 인코딩
            image_name_only = os.path.basename(image_name)
            for c, segm in enumerate(full_mask):
                rle = encode_mask_to_rle(segm)
                class_name = Config.CLASSES[c]
                results_dict[f"{class_name}_{image_name_only}"] = rle

    # 4. sample_submission.csv와 매칭
    print("Post-processing and Matching with sample_submission.csv...")
    sample_sub_path = "sample_submission.csv"
    
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        
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

    # 5. 최종 저장
    save_path = f"submission_{Config.EXPERIMENT_NAME}.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == '__main__':
    test()
