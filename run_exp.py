import sys
import os
import importlib

# 각 모듈에서 실행 함수 가져오기
from train import train
from config import Config

def main():
    print(f"=======================================================")
    print(f" [RUN ALL] Experiment Name: {Config.EXPERIMENT_NAME}")
    print(f"=======================================================")

    # 1. 학습 시작
    print("\n>>> [Stage 1] Start Training...")
    try:
        train() # train.py의 train() 함수 실행
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

if __name__ == '__main__':
    main()