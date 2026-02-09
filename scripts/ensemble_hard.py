import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tqdm import tqdm
from glob import glob
from pathlib import Path
from config import Config
from utils import decode_rle_to_mask, encode_mask_to_rle


def ensemble_binary_maps(binary_map_dirs, output_dir="./", create_csv=True, vote_threshold=None):
    """
    바이너리 맵들을 하드보팅 앙상블하고 RLE CSV 생성
    
    Args:
        binary_map_dirs: 바이너리 맵이 저장된 디렉토리 리스트
        output_dir: 앙상블 결과 저장 경로 (기본값: "./")
        create_csv: RLE CSV 파일 생성 여부
        vote_threshold: 투표 임계값 (None이면 과반수)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 개수 및 투표 임계값 설정
    num_models = len(binary_map_dirs)
    if vote_threshold is None:
        vote_threshold = num_models // 2
    
    # 모델 이름 추출 (디렉토리 basename)
    model_names = [Path(d).name[:7] for d in binary_map_dirs]
    
    # 첫 번째 디렉토리에서 파일 리스트 가져오기
    first_dir_files = sorted(glob(os.path.join(binary_map_dirs[0], "*.npz")))
    base_names = [os.path.basename(f) for f in first_dir_files]
    
    print(f"Ensemble {num_models} models: {', '.join(model_names)}")
    print(f"Vote threshold: > {vote_threshold} votes")
    print(f"Total images: {len(base_names)}")
    
    # 클래스 이름
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19',
        'Trapezium', 'Trapezoid', 'Capitate', 'Hamate',
        'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform',
        'Radius', 'Ulna'
    ]
    
    # RLE 결과 저장용
    rle_results = []
    
    # 앙상블 수행
    for base_name in tqdm(base_names, desc="Ensembling"):
        masks = []
        
        # 각 모델의 바이너리 맵 로드
        for model_dir in binary_map_dirs:
            npz_path = os.path.join(model_dir, base_name)
            
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} not found")
                break
            
            with np.load(npz_path) as data:
                mask = data['mask']
                masks.append(mask)
        
        # 모든 모델의 마스크가 있는지 확인
        if len(masks) != num_models:
            print(f"Skipping {base_name}: incomplete masks")
            continue
        
        # Stack & 하드보팅
        stacked_masks = np.stack(masks, axis=0)  # (num_models, 29, H, W)
        vote_counts = np.sum(stacked_masks, axis=0)  # (29, H, W)
        ensemble_mask = (vote_counts > vote_threshold).astype(np.uint8)  # (29, H, W)
        
        # RLE 인코딩 (CSV 생성 시)
        if create_csv:
            image_name = base_name.replace('.npz', '.png')
            
            for class_idx, class_name in enumerate(CLASSES):
                class_mask = ensemble_mask[class_idx]
                rle = encode_mask_to_rle(class_mask)
                rle_results.append({
                    'image_name': image_name,
                    'class': class_name,
                    'rle': rle
                })
    
    # CSV 저장
    if create_csv and len(rle_results) > 0:
        df = pd.DataFrame(rle_results)
        
        # 파일명 생성: ensemble_모델1_모델2_모델3.csv
        output_filename = f"ensemble_{'_'.join(model_names)}.csv"
        csv_path = os.path.join(output_dir, output_filename)
        
        df.to_csv(csv_path, index=False)
        print(f"\n✓ RLE CSV saved to: {csv_path}")
        print(f"  - Total predictions: {len(rle_results)}")
        print(f"  - Images: {len(rle_results) // len(CLASSES)}")
    else:
        print("\nNo results to save!")


if __name__ == '__main__':
    # 앙상블할 바이너리 맵 디렉토리들
    binary_map_dirs = [
        "bi_map/KJE_014",
        "bi_map/WJH_037_hrnet_w48_1024_focal_dice_sw",
        "bi_map/KKM_010_deeplabv3_sliding",
    ]
    
    # 앙상블 실행
    # 결과: ./ensemble_KJE_014_WJH_037_exp3.csv
    ensemble_binary_maps(
        binary_map_dirs=binary_map_dirs,
        output_dir="./",
        create_csv=True,
        vote_threshold=None  # None이면 과반수 (2개 이상)
    )
    
    print("\n✓ Ensemble completed!")
