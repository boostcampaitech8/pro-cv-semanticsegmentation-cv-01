import sys
import os
import importlib

# 각 모듈에서 실행 함수 가져오기
# [Modified] Dynamic Import based on Dataset Type
from config import Config

def get_trainer():
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
        if hasattr(dataset_module, 'get_dali_loader'):
            print(f">> DALI Dataset detected ({Config.DATASET_FILE}). Using train_dali.py")
            from train_dali import train
            return train
        else:
            print(f">> Standard Dataset detected ({Config.DATASET_FILE}). Using train.py")
            from train import train
            return train
    except Exception as e:
        print(f"Warning: Could not check dataset type ({e}). Defaulting to train.py")
        from train import train
        return train

def main():
    print(f"=======================================================")
    print(f" [RUN ALL] Experiment Name: {Config.EXPERIMENT_NAME}")
    print(f"=======================================================")

    # # 1. 학습 시작
    print("\n>>> [Stage 1] Start Training...")
    try:
        train_func = get_trainer() # [Modified]
        train_func() 
        print(">>> [Stage 1] Training Completed Successfully.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1) # 학습 실패 시 프로그램 종료

    # 2. 추론 시작
    print("\n>>> [Stage 2] Start Inference...")
    try:
        inference_module = importlib.import_module(Config.INFERENCE_FILE)
        
        # 불러온 파일 안에서 'test' 라는 이름의 함수를 찾습니다.
        test_func = getattr(inference_module, 'test')
        
        # 함수 실행
        test_func()
        print(">>> [Stage 2] Inference Completed Successfully.")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        sys.exit(1)

    print(f"\nAll processes finished. Check submission_{Config.EXPERIMENT_NAME}.csv")

# run_exp.py 하단 수정

if __name__ == '__main__':
    import argparse

    # 1. 인자 파서 생성
    parser = argparse.ArgumentParser()
    
    # 바꾸고 싶은 설정들만 argument로 추가
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--dataset_file', type=str, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--resize_size', type=int, nargs='+', default=None, help="Input one int (square) or two ints (H W)")
    
    args = parser.parse_args()

    # 2. Config 덮어쓰기 (입력된 값만)
    if args.exp_name: 
        Config.EXPERIMENT_NAME = args.exp_name
        Config.WANDB_RUN_NAME = args.exp_name # WandB 이름도 같이 변경
        Config.SAVED_DIR = os.path.join("checkpoints", args.exp_name) # 저장 경로도 변경
        if not os.path.exists(Config.SAVED_DIR):
            os.makedirs(Config.SAVED_DIR)

    if args.model_file: Config.MODEL_FILE = args.model_file
    if args.dataset_file: Config.DATASET_FILE = args.dataset_file
    if args.loss: Config.LOSS_FUNCTION = args.loss
    if args.lr: Config.LR = args.lr
    if args.epoch: Config.NUM_EPOCHS = args.epoch
    if args.resize_size:
        # 숫자가 1개만 들어오면 (예: 512) -> (512, 512)로 변환
        if len(args.resize_size) == 1:
            size = args.resize_size[0]
            Config.RESIZE_SIZE = (size, size)
        # 숫자가 2개 들어오면 (예: 512 1024) -> 그대로 튜플로 변환
        elif len(args.resize_size) == 2:
            Config.RESIZE_SIZE = tuple(args.resize_size)
        else:
            raise ValueError("resize_size는 숫자 1개 또는 2개만 입력 가능합니다.")

    # 3. 메인 실행
    main()