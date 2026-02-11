import os


class Config:
    EXPERIMENT_NAME = "WJH_073_ensemble_sliding_Gaussian"

    USE_WANDB = True  # True: 사용 / False: 사용 안 함 (디버깅 등)
    WANDB_ENTITY = "ckgqf1313-boostcamp"
    WANDB_PROJECT = "HandBoneSeg"  # 프로젝트 이름
    WANDB_RUN_NAME = EXPERIMENT_NAME  # 실험 이름을 Run 이름으로 사용

    # [1] 파일 선택
    DATASET_FILE = "dataset.dataset_dali_sliding_exclude"
    MODEL_FILE = "model.model_nnunet"
    INFERENCE_FILE = "inference.inference_sliding"

    # [Sliding Window 설정]
    WINDOW_SIZE = 1024  # 윈도우 크기
    STRIDE = 512 # 스트라이드 (2x2 패치)

    # [2] 학습 환경
    DATA_ROOT = "../data"
    IMAGE_ROOT = os.path.join(DATA_ROOT, "train/DCM")
    LABEL_ROOT = os.path.join(DATA_ROOT, "train/outputs_json")
    TEST_IMAGE_ROOT = os.path.join(DATA_ROOT, "test/DCM")

    SAVED_DIR = os.path.join("checkpoints", EXPERIMENT_NAME)
    if not os.path.exists(SAVED_DIR):
        os.makedirs(SAVED_DIR)

    RESIZE_SIZE = (1024, 1024)  # DALI sliding에서는 무시됨 (원본 2048 유지)
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    NUM_EPOCHS = 100

    # =================================================================
    # [맵 생성 설정]
    # =================================================================
    MAP_MODEL = "best"  # 'best', 'final', 'finetuned'
    # =================================================================

    # [2] 학습 제어 설정 (NEW)
    # ========================================================
    USE_EARLY_STOPPING = (
        True  # True: 성능 향상 없으면 조기 종료 / False: 무조건 끝까지 학습
    )
    EARLY_STOPPING_PATIENCE = 5  # 몇 번 참을지
    EARLY_STOPPING_MIN_DELTA = 0.0000  # 이만큼 올라야 오른걸로 치겠다

    SAVE_BEST_MODEL = (
        True  # True: 최고 점수 갱신 시 저장 / False: 저장 안 함 (마지막 모델만 남음)
    )
    # ========================================================

    VAL_EVERY = 1  # 몇 Epoch마다 검증할지
    WANDB_VIS_EVERY = 5

    LR = 8e-5
    RANDOM_SEED = 21
    OPTIMIZER = "AdamW"

    # [NEW] Scheduler 설정
    # 옵션: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR', 'None'
    SCHEDULER = "CosineAnnealingLR"

    # 1. ReduceLROnPlateau 설정 (성능 향상 멈추면 줄이기 - 추천)
    SCHEDULER_PATIENCE = 2  # 성능 향상 없는 Epoch 수
    SCHEDULER_FACTOR = 0.5  # 줄이는 비율 (0.5면 반토막)
    SCHEDULER_MIN_LR = 1e-6  # 최소 LR (이 밑으로는 안 줄임)

    # 2. StepLR 설정 (일정 Epoch마다 무조건 줄이기)
    SCHEDULER_STEP_SIZE = 10  # 10 Epoch마다
    SCHEDULER_GAMMA = 0.5  # 절반으로 줄임

    # 3. CosineAnnealingLR 설정 (부드럽게 줄였다 늘렸다 - 고급)
    SCHEDULER_T_MAX = 40  # 보통 총 Epoch 수와 맞춤

    # [NEW] Warmup Scheduler 설정 (초반 발산 방지)
    USE_WARMUP = True  # True: 사용 / False: 사용 안 함
    WARMUP_EPOCHS = 5  # 초반 몇 Epoch 동안 Warmup 할지
    WARMUP_MIN_LR = 1e-6  # Warmup 시작 LR (여기서부터 목표 LR까지 증가)

    # [TTA 설정] - inference_tta 사용 시 적용
    # [Optimized] Sigma=1.5, Scale=[1.0], Optimal Thresholds
    TTA_MODE = "" 
    TTA_SCALES = [1.0]
    USE_SLIDING_WINDOW = True 
    USE_GAUSSIAN_SLIDING = True 
    GAUSSIAN_SIGMA = 0.3
    USE_OPTIMAL_THRESHOLD = False

    # [Optimized via tools/find_best_params.py]
    OPTIMAL_THRESHOLDS = {
        'finger-1': 0.30,
        'finger-2': 0.40,
        'finger-3': 0.45,
        'finger-4': 0.35,
        'finger-5': 0.30,
        'finger-6': 0.30,
        'finger-7': 0.45,
        'finger-8': 0.35,
        'finger-9': 0.30,
        'finger-10': 0.35,
        'finger-11': 0.50,
        'finger-12': 0.30,
        'finger-13': 0.35,
        'finger-14': 0.40,
        'finger-15': 0.45,
        'finger-16': 0.55,
        'finger-17': 0.55,
        'finger-18': 0.50,
        'finger-19': 0.45,
        'Trapezium': 0.55,
        'Trapezoid': 0.45,
        'Capitate': 0.55,
        'Hamate': 0.40,
        'Scaphoid': 0.40,
        'Lunate': 0.50,
        'Triquetrum': 0.45,
        'Pisiform': 0.40,
        'Radius': 0.45,
        'Ulna': 0.45,
    }

    # [앙상블 설정] - inference_ensemble 사용 시 적용
    # 각 모델별로 TTA, Sliding 여부를 다르게 설정 가능
    # 각 모델별로 어떤 추론 스크립트를 쓸지 지정 가능
    ENSEMBLE_MODELS = [

         {
            'path': "ensemble/best_model_hrnet.pt", 
            'inference_file': 'inference.inference_tta', # TTA (w/ Sliding) 사용
            'dataset_file': 'dataset.dataset_dali_sliding_exclude',
            'resize_size': (2048, 2048),
            'window_size': 1024,
            'stride': 512
         },

          {
             'path': "ensemble/best_model_nnunet.pt", 
             'inference_file': 'inference.inference_tta', # TTA (w/ Sliding) 사용
             'dataset_file': 'dataset.dataset_dali_sliding_exclude',
             'resize_size': (2048, 2048),
             'window_size': 1024,
             'stride': 512
          },

          {
             'path': "ensemble/best_model_deeplabv3.pt", 
             'inference_file': 'inference.inference_tta', # TTA (w/ Sliding) 사용
             'dataset_file': 'dataset.dataset_dali_sliding_exclude',
             'resize_size': (2048, 2048),
             'window_size': 1024,
             'stride': 512
          },
    ]
    # ENSEMBLE_STRATEGY removed (Managed by USE_OPTIMIZATION)
    ENSEMBLE_USE_OPTIMIZATION = (
        False  # True: 최적 가중치 자동 탐색 (Weighted Search) / False: 수동 or 균등
    )


    # [NEW] 가중치 최적화 방식
    # 'global' : 모든 클래스에 동일한 가중치 적용 (기존 방식)
    # 'class'  : 각 클래스별로 최적 가중치 따로 계산 (성능 더 좋음)
    ENSEMBLE_WEIGHT_METHOD = "global"

    # [NEW] 최적화 기준 (Metric)
    # 'soft' : Adam 사용. 미분 가능한 Soft Dice 최적화 (빠름, 일반적으로 권장)
    # 'hard' : Nelder-Mead 사용. 실제 평가 지표인 Hard Dice(mDice) 직접 최적화 (느림, 정확도 중시)
    ENSEMBLE_OPTIM_METRIC = "hard"

    # [NEW] 최적화 해상도 (Resolution)
    # 1024 : 메모리 절약 (기본) - 약간의 정확도 하락 감수
    # 2048 : 원본 해상도 - 모델 2~3개 이하일 때 권장 (메모리 많이 씀)
    ENSEMBLE_OPTIM_SIZE = 2048

    # [수동 가중치 설정] (USE_OPTIMIZATION = False 일 때 사용)
    # 모델 개수만큼 리스트로 입력해주세요. (합이 1이 되도록 권장)
    # 예: [0.7, 0.3] -> 첫 번째 모델에 70%, 두 번째에 30% 반영
    # None으로 두면 자동으로 1/N (균등) 적용됩니다.
    ENSEMBLE_WEIGHTS = None

    # [NEW] MAP Model
    # 'best' : 최고 모델
    # 'last' : 마지막 모델
    # 'finetuned' : fine-tuned 모델
    MAP_MODEL = 'best'
    
    # [Loss 선택지]

    # [Loss 선택지]
    # 1. 'BCE'               : 가장 기초. (지금 쓰고 계신 것)
    # 2. 'Focal'             : 뼈처럼 작고 어려운 객체 잡을 때 좋음.
    # 3. 'Combined_BCE_Dice' : 학습 안정성 + 성능 밸런스형 (추천)
    # 4. 'Combined_Focal_Dice': 캐글 등 상위 랭커들이 가장 많이 쓰는 조합 (강력 추천)
    # 5. 'Combined_Focal_Dice_Overlap': Focal + Dice + Overlap Penalty
    # util.py 참고
    LOSS_FUNCTION = "Dice"

    # [비율 설정] (앞쪽 Loss, 뒤쪽 Loss)
    # 예: (0.5, 0.5) -> 반반 (기본값)
    # 예: (0.3, 0.7) -> Dice(모양)에 더 집중
    LOSS_WEIGHTS = (0.4, 0.4, 0.2)

    # 1. Focal Loss 파라미터
    FOCAL_ALPHA = 0.5  # 클래스 불균형 해소 가중치
    FOCAL_GAMMA = 2.0  # 어려운 샘플에 대한 집중도

    # 2. Jaccard & Dice & Tversky 공용 Smooth
    LOSS_SMOOTH = 1.0

    # 3. Tversky Loss 파라미터 (FP vs FN 가중치)
    TVERSKY_ALPHA = 0.5  # FP(거짓 양성) 벌점
    TVERSKY_BETA = 0.5  # FN(못 찾은 뼈) 벌점 (0.7 등으로 올리면 Recall 상승)

    # 4. Generalized Dice Loss 파라미터
    GDL_SMOOTH = 1e-6
    GDL_GAMMA = 2.0

    # 5. Pixel Weighted BCE 파라미터
    PW_BCE_SMOOTH = 1e-6

    CLASSES = [
        "finger-1",
        "finger-2",
        "finger-3",
        "finger-4",
        "finger-5",
        "finger-6",
        "finger-7",
        "finger-8",
        "finger-9",
        "finger-10",
        "finger-11",
        "finger-12",
        "finger-13",
        "finger-14",
        "finger-15",
        "finger-16",
        "finger-17",
        "finger-18",
        "finger-19",
        "Trapezium",
        "Trapezoid",
        "Capitate",
        "Hamate",
        "Scaphoid",
        "Lunate",
        "Triquetrum",
        "Pisiform",
        "Radius",
        "Ulna",
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    # Finetuning (train+val) : 체크포인트 불러와서 full dataset으로 학습 -> val 없으므로 ReduceLROnPlateau 불가. 다른 scheduler 사용.
    USE_FINETUNE = False
