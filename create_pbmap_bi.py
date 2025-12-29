import torch
import numpy as np
import os
import importlib
import shutil
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from config import Config

def load_model(map_model):
    """MAP_MODEL 설정에 따라 모델 로드"""
    model_files = {
        'best': 'best_model.pt',
        'last': 'final_model.pt',
        'finetuned': 'finetuned_model.pt'
    }
    
    model_path = os.path.join(Config.SAVED_DIR, model_files[map_model])
    print(f"Loading model from: {model_path}")
    
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()
    return model

def get_dir_size(path):
    """디렉토리 전체 용량 계산 (bytes)"""
    if not os.path.exists(path):
        return 0
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def format_size(bytes):
    """bytes를 읽기 좋은 단위로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def process_in_chunks(model, test_dataset, inference_module, output_dir, chunk_size=50):
    """데이터셋을 청크로 나눠서 처리"""
    
    total_images = len(test_dataset)
    num_chunks = (total_images + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_images} images in {num_chunks} chunks (chunk_size={chunk_size})")
    
    total_temp_size = 0
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_images)
        
        print(f"\n{'='*60}")
        print(f"📦 Chunk {chunk_idx + 1}/{num_chunks}: Processing images {chunk_start+1}-{chunk_end}")
        print(f"{'='*60}")
        
        # 청크용 Subset 생성
        chunk_indices = list(range(chunk_start, chunk_end))
        chunk_dataset = Subset(test_dataset, chunk_indices)
        chunk_loader = DataLoader(
            chunk_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS
        )
        
        # Temp dir
        temp_dir = f"temp_bi_map_chunk_{chunk_idx}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Prepare config dict
            inference_cfg = {
                'tta_mode': Config.TTA_MODE,
                'scales': Config.TTA_SCALES,
                'window_size': getattr(Config, 'WINDOW_SIZE', None),
                'stride': getattr(Config, 'STRIDE', None),
            }
            
            # get_probs 호출 (이 청크만)
            print("Running inference...")
            inference_module.get_probs(model, chunk_loader, save_dir=temp_dir, **inference_cfg)
            
            # Temp 용량 측정
            temp_size = get_dir_size(temp_dir)
            total_temp_size += temp_size
            print(f"📊 Chunk temp size (float16): {format_size(temp_size)}")
            
            # 변환 전 output 용량 측정
            before_convert_size = get_dir_size(output_dir)
            
            # 변환 & 저장
            print("Converting to binary maps...")
            chunk_filenames = [os.path.basename(test_dataset.filenames[i]) for i in chunk_indices]
            
            for img_name in tqdm(chunk_filenames, desc=f"Chunk {chunk_idx + 1}"):
                base_name = os.path.splitext(img_name)[0]
                npy_path = os.path.join(temp_dir, f"{img_name}.npy")
                
                # Load probability map
                prob = np.load(npy_path).astype(np.float32)
                
                # Convert to binary
                binary_map = (prob > 0.5).astype(np.uint8)
                
                # Save compressed
                save_path = os.path.join(output_dir, f"{base_name}.npz")
                np.savez_compressed(save_path, mask=binary_map)
                
                # Remove temp file (안전하게)
                try:
                    os.remove(npy_path)
                except FileNotFoundError:
                    pass
            
            # 변환 후 output 용량 측정
            after_convert_size = get_dir_size(output_dir)
            chunk_converted_size = after_convert_size - before_convert_size
            
            print(f"💾 Chunk converted size (uint8 compressed): {format_size(chunk_converted_size)}")
            if chunk_converted_size > 0:
                print(f"🗜️  Chunk compression ratio: {temp_size / chunk_converted_size:.2f}x")
            print(f"📁 Total accumulated size: {format_size(after_convert_size)}")
            
        finally:
            # 에러 발생해도 반드시 temp 디렉토리 삭제
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🗑️  Cleaned up {temp_dir}")
        
        torch.cuda.empty_cache()
        print(f"✅ Chunk {chunk_idx + 1}/{num_chunks} complete")
    
    return total_temp_size

def main():
    # Setup
    output_dir = os.path.join("bi_map", Config.EXPERIMENT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 기존 temp 디렉토리들 정리 (이전 실행에서 남은 것들)
    print("Cleaning up old temp directories...")
    for item in os.listdir('.'):
        if item.startswith('temp_bi_map_chunk_'):
            shutil.rmtree(item)
            print(f"🗑️  Removed old {item}")
    
    # Load Model
    model = load_model(Config.MAP_MODEL)
    
    # Load Dataset & Inference Module
    dataset_module = importlib.import_module(Config.DATASET_FILE)
    inference_module = importlib.import_module(Config.INFERENCE_FILE)
    
    # Prepare Test Dataset
    XRayInferenceDataset = dataset_module.XRayInferenceDataset
    get_transforms = dataset_module.get_transforms
    test_dataset = XRayInferenceDataset(transforms=get_transforms(is_train=False))
    
    print(f"Total test images: {len(test_dataset)}")
    
    # 청크 단위로 처리 (기본값: 50개씩)
    chunk_size = 50
    total_temp_size = process_in_chunks(
        model, test_dataset, inference_module, output_dir, chunk_size=chunk_size
    )
    
    # 최종 통계
    final_size = get_dir_size(output_dir)
    avg_size = final_size / len(test_dataset) if test_dataset else 0
    
    print(f"\n{'='*60}")
    print(f"🎉 ALL COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Saved {len(test_dataset)} binary maps to {output_dir}")
    print(f"📦 Final size (compressed uint8 .npz): {format_size(final_size)}")
    print(f"📄 Average size per file: {format_size(avg_size)}")
    print(f"📊 Total temp size used: {format_size(total_temp_size)}")
    if final_size > 0:
        print(f"🗜️  Overall compression ratio: {total_temp_size / final_size:.2f}x")
    print(f"💾 Peak disk usage per chunk: ~{format_size(total_temp_size / (len(test_dataset) / chunk_size))}")

if __name__ == '__main__':
    main()


