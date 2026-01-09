import sys
import os

# 각 모듈에서 실행 함수 가져오기
from train import train
from inference import test
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
        test() # inference.py의 test() 함수 실행
        print(">>> [Stage 2] Inference Completed Successfully.")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        sys.exit(1)

    print(f"\nAll processes finished. Check submission_{Config.EXPERIMENT_NAME}.csv")

if __name__ == '__main__':
    main()