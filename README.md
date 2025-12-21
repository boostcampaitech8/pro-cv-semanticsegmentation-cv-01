    ├── dataset/             # 데이터셋 로드 및 전처리 모듈
    │   ├── dataset.py          # 기본 데이터셋
    │   ├── dataset_exclude.py  # Artifact 제거 데이터셋
    │   └── dataset_xnormalize.py # 정규화 제거 데이터셋
    ├── eda/                 # 탐색적 데이터 분석
    │   ├── EDA.ipynb           # 기본 EDA
    │   ├── eda_meta.ipynb       # 메타데이터 분석
    │   └── fiftyone.ipynb       # Fiftyone 시각화
    ├── model/               # 모델 정의 모듈
    │   ├── model_unet.py        # UNet 기본
    │   ├── model_unet++.py      # UNet++
    │   ├── model_fcn.py         # FCN
    │   ├── model_deeplabv3plus.py # DeepLabV3+
    │   ├── model_manet.py       # MAnet
    │   └── model_segformer.py   # SegFormer
    ├── inference/           # 추론 로직 모듈
    │   └── inference.py        # 추론 실행, 이후 TTA, window slide 등 다른 버전들 추가
    ├── config.py            # [Control Center] 모든 실험 하이퍼파라미터 및 경로 설정
    ├── run_exp.py           # [Main Executor] 학습부터 추론까지 한 번에 실행 (Dynamic Loading + CLI Args)
    ├── schedule.py          # 학습 스케줄링 관련
    ├── train.py             # 학습 루프 (Validation 및 Model Saving)
    ├── utils.py             # Dice Score, RLE Encoding 등 유틸리티
    ├── visualize.py         # 시각화 도구
    └── sample_submission.csv
