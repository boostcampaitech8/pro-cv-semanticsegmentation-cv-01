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

    # 2. 데이터 로더
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # 결과를 저장할 딕셔너리 (Key: "클래스_파일명", Value: RLE)
    results_dict = {}
    
    # 3. 인퍼런스 시작
    print("Start Inference...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader):
            images = images.cuda()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['out']
            
            # 원본 사이즈(2048, 2048)로 복원
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).detach().cpu().numpy()
            
            for output, image_path in zip(outputs, image_names):
                # [핵심 수정] 경로를 제외한 순수 파일명만 추출 (예: test/001.png -> 001.png)
                image_name = os.path.basename(image_path)
                
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    class_name = Config.CLASSES[c]
                    # 나중에 매칭을 위해 딕셔너리에 저장
                    results_dict[f"{class_name}_{image_name}"] = rle

    # 4. 베이스라인 스타일로 sample_submission.csv와 매칭
    print("Post-processing and Matching with sample_submission.csv...")
    sample_sub_path = "sample_submission.csv"
    
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        
        # sample_submission의 순서에 맞춰 RLE 값 채우기
        final_rles = []
        for _, row in sample_df.iterrows():
            # sample_df에 있는 이름과 우리가 저장한 이름을 매칭
            key = f"{row['class']}_{row['image_name']}"
            # 매칭되는 결과가 없으면 빈 문자열("")을 넣음
            final_rles.append(results_dict.get(key, ""))
            
        sample_df['rle'] = final_rles
        final_df = sample_df
    else:
        print("Warning: sample_submission.csv not found!")
        # 대체용 데이터프레임 생성
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