# 🦴 Hand Bone Semantic Segmentation (CV-01)

본 프로젝트는 손 엑스레이(Hand X-ray) 영상에서 29종의 주요 본(Bone) 영역을 정밀하게 분할하는 의료 영상 세그멘테이션 프로젝트입니다. NVIDIA DALI를 통한 데이터 로딩 가속과 해부학적 특성을 반영한 전처리 파이프라인을 특징으로 합니다.

---

## 🛠 환경 설정

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

## 📂 디렉토리 구조 (Directory Structure)

```text
.
├── config.py            # [Control Center] 모든 실험 설정 및 하이퍼파라미터 중심 관리
├── run_exp.py           # [Unified] 통합 실행 엔진 (CLI 인자 & 백그라운드 모드 지원)
├── train.py             # 기본 PyTorch 학습 코어
├── train_dali.py        # [High-Speed] NVIDIA DALI 기반 가속 학습 엔진
├── utils.py             # [Common] 시드 고정, RLE 인코딩, Custom Loss (한글화 완료)
│
├── scripts/             # [Utility] 독립 실행형 스크립트 모음
│   ├── schedule.py          # 실험 예약 자동화 (다중 실험 순차 실행)
│   ├── ensemble_hard.py     # 앙상블 (Hard Voting) 도구
│   ├── create_pbmap_bi.py   # 확률 맵 생성 및 바이너리 변환
│   ├── denoise_csv.py       # CSV 결과 노이즈 제거 및 후처리
│   ├── preprocess_to_jpeg.py # DALI 로딩용 JPEG 사전 변환 도구
│   └── visualize_csv.py     # CSV 기반 예측 결과 시각화
├── eda/                 # [Analysis] 데이터 분석 및 시각화 노트북 (Jupyter)
├── dataset/             # 데이터셋 로더 및 전처리 모듈 (DALI/Sliding Window 등)
├── model/               # 다양한 모델 정의 (nnUNet, SegFormer 등 20+ 지원)
├── inference/           # 추론 파이프라인 및 TTA 설정
├── data/                # 데이터 참조 파일 (sample_submission.csv 등)
└── checkpoints/         # 모델 가중치 저장소
```

---

## 🚀 프로젝트 핵심 기능

### ⚡ 1. 통합 실행 엔진 (`run_exp.py`)
- **데이터셋 자동 감지**: 선택된 데이터셋 모듈에 따라 DALI 학습(`train_dali.py`) 또는 일반 학습(`train.py`)으로 자동 분기합니다.
- **설정 우선순위**:
    1. **CLI Arguments (최우선)**: `python run_exp.py --lr 1e-4`와 같이 실행 시 인자를 주면 `config.py` 내용을 덮어씁니다.
    2. **Config File**: 중앙 제어 파일(`config.py`)의 설정값이 기본으로 사용됩니다.
- **백그라운드 지원**: `--bg` 옵션을 통해 서버 연결이 끊겨도 `nohup` 기반으로 안전하게 학습을 지속할 수 있습니다.

### 🍱 2. NVIDIA DALI 기반 데이터 가속
- 고해상도(2048x2048) 이미지의 디코딩 및 증강을 GPU에서 처리하여 병목을 제거했습니다.
- **Hybrid JPEG Pipeline**: `scripts/preprocess_to_jpeg.py`를 통한 사전 변환과 CLAHE 연산을 결합하여 학습 효율을 극대화했습니다.

---

## 📖 사용 방법

### 1. 설정 변경 (`config.py`)
중앙 관리 파일에서 모델, 데이터셋, 학습률 등을 설정합니다.

**주요 설정 항목:**
- `MODEL_FILE`: 사용할 모델 (`model.model_nnunet`, `model.model_segformer` 등)
- `DATASET_FILE`: 데이터셋 로더 선택
- `EXPERIMENT_NAME`: 실험 이름 (체크포인트 폴더명 및 WandB 로그명)
- `BATCH_SIZE`: 배치 크기
- `NUM_EPOCHS`: 학습 에폭 수

**예시:**
```python
MODEL_FILE = 'model.model_unet'
DATASET_FILE = 'dataset.dataset_dali_sliding_exclude' 
EXPERIMENT_NAME = 'My_First_Experiment'
BATCH_SIZE = 4
```

### 2. 학습 및 추론 실행
```bash
# 기본 실행 (config.py 설정 직접 반영)
python run_exp.py

# CLI 인자로 특정 설정만 바꿔서 실행 (가장 추천하는 방식)
python run_exp.py --exp_name New_Trial --epoch 50 --lr 0.0001 --batch_size 4

# 백그라운드에서 실행 (자체 --bg 옵션 사용)
python run_exp.py --exp_name My_Trial --bg
```

### 3. 추론만 실행 (학습된 모델 사용)
```bash
# 기본 추론 (설정된 Config에 따라 실행)
python inference/inference.py
```

### 4. 백그라운드 실행 및 로그 관리 (Advanced)
Linux 환경에서 직접 백그라운드로 실행하고 로그를 관리하는 방법입니다.
```bash
# nohup을 이용한 백그라운드 실행 (Config의 실험명 + 날짜/시간 사용)
EXP_NAME=$(python3 -c 'from config import Config; print(Config.EXPERIMENT_NAME)') && \
nohup python run_exp.py > ${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 실시간으로 로그 확인하기 (가장 최근 생성된 로그 파일)
tail -f $(ls -t *.log | head -n 1)
```

### 5. 다중 실험 자동화 (`scripts/schedule.py`)
여러 실험을 예약 리스트에 등록한 후 순차적으로 자동 실행합니다.
1. `scripts/schedule.py` 파일 내 `experiments` 리스트에 실험 설정 추가
2. 스크립트 실행:
```bash
python scripts/schedule.py
```

### 6. 결과 시각화 및 후처리
```bash
# 앙상블 결과 시각화
python scripts/visualize_csv.py --csv path/to/result.csv

# 결과 노이즈 제거 처리
python scripts/denoise_csv.py --input path/to/in.csv --output path/to/out.csv
```

---

## 👥 팀 정보
- **Team**: Boostcamp AI Tech 8기 CV-01 (Hand Segmentation)
- **Focus**: Precision Medical Image Segmentation
