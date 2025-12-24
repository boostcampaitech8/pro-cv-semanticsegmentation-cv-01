# 🦴 Hand Bone Semantic Segmentation (CV-01)

본 프로젝트는 손 엑스레이(Hand X-ray) 영상에서 29종의 주요 본(Bone) 영역을 정밀하게 분할하는 의료 영상 세그멘테이션 프로젝트입니다. NVIDIA DALI를 통한 데이터 로딩 가속과 해부학적 특성을 반영한 전처리 파이프라인을 특징으로 합니다.

---

## � 환경 설정

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt

# NVIDIA DALI 설치 (선택 사항, GPU 가속 데이터 로딩 사용 시)
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
```

### 2. 데이터 준비
데이터를 다음 구조로 배치하세요:
```
../data/
├── train/
│   ├── DCM/           # 학습 이미지 (.png)
│   └── outputs_json/  # 라벨 JSON 파일
└── test/
    └── DCM/           # 테스트 이미지 (.png)
```

---

## �📁 디렉토리 구조 (Directory Structure)

```text
.
├── checkpoints/         # 모델 가중치 저장 및 관리
│   └── Base_UNet/          # 방향 판별 모델 가중치 등
├── dataset/             # 데이터셋 로드 및 전처리 모듈
│   ├── dataset.py          # 기본 데이터셋 로더
│   ├── dataset_dali_v1.py  # [New] DALI + CPU SSR (안정성)
│   ├── dataset_dali_v2.py  # [New] DALI + GPU SSR (자동 강도 보정)
│   ├── dataset_crop.py     # BBox 기반 손 중심 크롭 (Hand-centered)
│   ├── dataset_flip.py     # 모델 기반 손 방향 정규화 (Flip)
│   ├── dataset_exclude.py  # Artifact(ID363, ID387) 제외 필터링
│   └── ... (dataset_clahe, dataset_final 등 실험용 로더 다수)
├── eda/                 # 탐색적 데이터 분석 (EDA)
│   ├── Crop_Hand_Forearm.ipynb # 손 vs 전완부 면적 및 크롭 전략 분석
│   ├── Hand_Direction_Analysis.ipynb # 손 방향(왼손/오른손) 판별 분석
│   ├── EDA_Img_processing.ipynb # 이미지 전처리 및 정렬 분석
│   ├── eda_meta.ipynb      # 환자 메타데이터 분석
│   ├── fiftyone.ipynb      # Fiftyone을 활용한 데이터 시각화
│   └── EDA.ipynb           # 기본 이미지 및 라벨 분석
├── model/               # 모델 정의 (Architectures)
│   ├── model_nnunet.py      # Main Model (Residual UNet)
│   ├── model_segformer.py   # Transformer-based Architecture
│   └── ... (UNet++, DeepLabV3+, MAnet, FCN 등 20+ 모델 지원)
├── inference/           # 추론 및 결과 생성
│   ├── inference.py        # 기본 추론 및 RLE 생성
│   ├── inference_crop.py   # 크롭 기반 추론 및 마스크 원복 로직
│   └── inference_flip.py   # 2단계 추론 (방향 판별 -> 정규화 -> 세그멘테이션)
├── config.py            # [Control Center] 모든 실험 설정 및 하이퍼파라미터
├── train_dali.py        # [New] NVIDIA DALI 기반 초고속 학습 엔진
├── run_exp.py           # [Unified] 통합 실행 스크립트 (DALI/PyTorch 자동 감지)
├── schedule.py          # [Scheduler] 다중 실험 예약 자동화
└── train.py             # 기존 PyTorch Learner
```

---

## 🚀 프로젝트 핵심 기능

### ⚡ 1. NVIDIA DALI 데이터 가속 (`train_dali.py`)
- **병목 해결**: 2048x2048 고해상도 이미지의 디코딩 및 증강을 GPU에서 처리하여 학습 속도를 획기적으로 개선했습니다.
- **주요 특징**: NVJPEG 기반 하드웨어 가속 디코딩, GPU 기반 실시간 Resize/Flip/Rotate 지원.

### 🍱 2. 데이터 전처리 전략 (Preprocessing)
- **Image Resizing**: 고해상도 이미지를 모델 입력을 위해 512x512 또는 1024x1024 등으로 리사이즈하여 사용합니다.
- **Contrast Enhancement (CLAHE)**: 뼈의 윤곽을 뚜렷하게 하기 위해 대비 제한 적응형 히스토그램 평활화(CLAHE)를 적용합니다.
- **Standard Augmentation**: Albumentations 라이브러리를 활용하여 Flip, Rotate, Brightness/Contrast 조정 등 모델의 일반화 성능을 높이기 위한 기본적인 증강 기법을 적용합니다.

---

## 🛠 사용 방법

### 1. 설정 변경 (`config.py`)
중앙 제어 파일에서 모델, 데이터셋, 하이퍼파라미터를 설정합니다.

**주요 설정 항목:**
- `MODEL_FILE`: 사용할 모델 (`model.model_nnunet`, `model.model_segformer` 등)
- `DATASET_FILE`: 데이터셋 로더 (`dataset.dataset`, `dataset.dataset_dali`, `dataset.dataset_clahe` 등)
- `EXPERIMENT_NAME`: 실험 이름 (체크포인트 폴더명 및 WandB 로그명)
- `BATCH_SIZE`: 배치 크기 (GPU 메모리에 따라 조정, 512x512 기준 8~16, 1024x1024 기준 2~4)
- `NUM_EPOCHS`: 학습 에폭 수
- `LR`: 학습률 (기본값: 1e-4)
- `LOSS_FUNCTION`: 손실 함수 (`BCE`, `Dice`, `Focal`, `Combined_BCE_Dice` 등)

**예시:**
```python
MODEL_FILE = 'model.model_nnunet'
DATASET_FILE = 'dataset.dataset_dali'  # DALI 사용 시
EXPERIMENT_NAME = 'nnUNet_DALI_Run'
BATCH_SIZE = 8
NUM_EPOCHS = 100
```

### 2. 기본 학습 실행
```bash
# 학습만 진행할 경우
python train.py

# 학습부터 추론 결과 CSV 생성까지 자동 실행
python run_exp.py --exp_name my_first_run --model_file model.model_nnunet
```

### 3. DALI 기반 고속 학습 실행
```bash
# 학습부터 추론 결과 CSV 생성까지 한 번에 실행 (GPU 가속 데이터 로딩)
python run_exp_dali.py --exp_name dali_test --model_file model.model_nnunet
```

### 4. 추론만 실행 (학습된 모델 사용)
```bash
# 기본 추론
python inference/inference.py
```

### 5. 백그라운드 실행 및 로그 저장 (Linux 명령어)
서버 접속이 끊겨도 학습이 유지되도록 하고, 모든 로그를 파일로 남기는 권장 방법입니다.
```bash
# nohup을 이용한 백그라운드 실행 (Config의 실험명 + 날짜/시간 사용)
EXP_NAME=$(python3 -c 'from config import Config; print(Config.EXPERIMENT_NAME)') && \
nohup python run_exp.py > ${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 실시간으로 로그 확인하기 (가장 최근 생성된 로그 파일)
tail -f $(ls -t *.log | head -n 1)
```

### 6. 다중 실험 자동화 (`schedule.py`)
여러 실험을 예약하여 순차적으로 실행할 수 있습니다.
1. `schedule.py` 파일 내 `experiments` 리스트에 실험 설정 추가
2. 스크립트 실행:
```bash
python schedule.py
```

---

## 👥 팀 정보
- **Team**: Boostcamp AI Tech 8기 CV-01 (Hand Segmentation)
- **Focus**: Precision Medical Image Segmentation
