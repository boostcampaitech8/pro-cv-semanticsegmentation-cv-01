    ├── dataset/             # 데이터셋 로드 및 전처리 모듈
    │   ├── dataset.py          # 기본
    │   └── dataset_exclude.py  # Artifact 제외
    ├── eda/                 # 탐색적 데이터 분석
    │   └── ...
    ├── model/               # 모델 정의 모듈
    │   ├── model_unet.py       # UNet
    │   ├── model_segformer.py  # SegFormer
    │   └── ...
    ├── inference/           # 추론 로직 모듈
    │   └── inference.py        # 기본, 이후 TTA, sliding window 등 추가
    ├── config.py            # [Control Center] 모든 실험 하이퍼파라미터 및 경로 설정
    ├── run_exp.py           # [Main Executor] 학습부터 추론까지 한 번에 실행 (Dynamic Loading)
    ├── train.py             # 학습 루프 (Validation 및 Model Saving)
    ├── utils.py             # Dice Score, RLE Encoding 등 유틸리티
    ├── visualize.py         # 시각화 도구
    └── sample_submission.csv
