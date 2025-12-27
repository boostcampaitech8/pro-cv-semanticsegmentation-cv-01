import os

class Config:
    EXPERIMENT_NAME = "WJH_035_hrnet_w48_1024_focal_dice_exclude"
    
    USE_WANDB = True             # True: 사용 / False: 사용 안 함 (디버깅 등)
    WANDB_ENTITY = "ckgqf1313-boostcamp"
    WANDB_PROJECT = "HandBoneSeg" # 프로젝트 이름
    WANDB_RUN_NAME = EXPERIMENT_NAME # 실험 이름을 Run 이름으로 사용

    # [1] 파일 선택
    DATASET_FILE = 'dataset.dataset_dali_exclude'
    MODEL_FILE = 'model.model_hrnet_w48'
    INFERENCE_FILE = 'inference.inference_TTA'
    
    # [Sliding Window 설정]
    WINDOW_SIZE = 1024  # 윈도우 크기
    STRIDE = 1024       # 스트라이드 (2x2 패치)
    
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
    NUM_WORKERS = 4
    NUM_EPOCHS = 200
    
    # [2] 학습 제어 설정 (NEW)
    # ========================================================
    USE_EARLY_STOPPING = True   # True: 성능 향상 없으면 조기 종료 / False: 무조건 끝까지 학습
    EARLY_STOPPING_PATIENCE = 5 # 몇 번 참을지
    EARLY_STOPPING_MIN_DELTA = 0.000 # 이만큼 올라야 오른걸로 치겠다
    
    SAVE_BEST_MODEL = True      # True: 최고 점수 갱신 시 저장 / False: 저장 안 함 (마지막 모델만 남음)
    # ========================================================

    VAL_EVERY = 1               # 몇 Epoch마다 검증할지
    WANDB_VIS_EVERY = 5
    
    LR = 5e-5
    RANDOM_SEED = 21
    OPTIMIZER = 'AdamW'

    # [NEW] Scheduler 설정
    # 옵션: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR', 'None'
    SCHEDULER = 'CosineAnnealingLR' 
    
    # 1. ReduceLROnPlateau 설정 (성능 향상 멈추면 줄이기 - 추천)
    SCHEDULER_PATIENCE = 2      # 성능 향상 없는 Epoch 수
    SCHEDULER_FACTOR = 0.5      # 줄이는 비율 (0.5면 반토막)
    SCHEDULER_MIN_LR = 1e-6     # 최소 LR (이 밑으로는 안 줄임)
    
    # 2. StepLR 설정 (일정 Epoch마다 무조건 줄이기)
    SCHEDULER_STEP_SIZE = 10    # 10 Epoch마다
    SCHEDULER_GAMMA = 0.5       # 절반으로 줄임
    
    # 3. CosineAnnealingLR 설정 (부드럽게 줄였다 늘렸다 - 고급)
    SCHEDULER_T_MAX = 30        # 보통 총 Epoch 수와 맞춤

    # [NEW] Warmup Scheduler 설정 (초반 발산 방지)
    USE_WARMUP = True           # True: 사용 / False: 사용 안 함
    WARMUP_EPOCHS = 5           # 초반 몇 Epoch 동안 Warmup 할지
    WARMUP_MIN_LR = 1e-6        # Warmup 시작 LR (여기서부터 목표 LR까지 증가)
    
    # [TTA 설정] - inference_tta 사용 시 적용
    TTA_MODE = '' # 'hflip', 'vflip', 'd4'
    TTA_SCALES = [0.9, 1.0, 1.5] # [0.75, 1.0, 1.25] 등 멀티스케일 설정 가능

    # [앙상블 설정] - inference_ensemble 사용 시 적용
    # 각 모델별로 TTA, Sliding 여부를 다르게 설정 가능
    # 각 모델별로 어떤 추론 스크립트를 쓸지 지정 가능
    ENSEMBLE_MODELS = [
        # {
        #    'path': "saved/WJH_023_hrnet/best_model.pt", 
        #    'inference_file': 'inference.inference_tta',  # 사용할 스크립트 지정
        #    'tta_mode': 'hflip', 
        #    'scales': [1.0]
        # },
        # {
        #    'path': "saved/WJH_024_hrnet_ocr/best_model.pt", 
        #    'inference_file': 'inference.inference_sliding', # 슬라이딩 윈도우 스크립트 지정
        #    'window_size': 1024,
        #    'stride': 1024
        # }
    ]
    # ENSEMBLE_STRATEGY removed (Managed by USE_OPTIMIZATION)
    ENSEMBLE_USE_OPTIMIZATION = False # True: 최적 가중치 자동 탐색 (Weighted Search) / False: 수동 or 균등
    
    # [수동 가중치 설정] (USE_OPTIMIZATION = False 일 때 사용)
    # 모델 개수만큼 리스트로 입력해주세요. (합이 1이 되도록 권장)
    # 예: [0.7, 0.3] -> 첫 번째 모델에 70%, 두 번째에 30% 반영
    # None으로 두면 자동으로 1/N (균등) 적용됩니다.
    ENSEMBLE_WEIGHTS = None

    # [Loss 선택지]

    # [Loss 선택지]
    # 1. 'BCE'               : 가장 기초. (지금 쓰고 계신 것)
    # 2. 'Focal'             : 뼈처럼 작고 어려운 객체 잡을 때 좋음.
    # 3. 'Combined_BCE_Dice' : 학습 안정성 + 성능 밸런스형 (추천)
    # 4. 'Combined_Focal_Dice': 캐글 등 상위 랭커들이 가장 많이 쓰는 조합 (강력 추천)
    # util.py 참고
    LOSS_FUNCTION = 'Combined_Focal_Dice'

    # [비율 설정] (앞쪽 Loss, 뒤쪽 Loss)
    # 예: (0.5, 0.5) -> 반반 (기본값)
    # 예: (0.3, 0.7) -> Dice(모양)에 더 집중
    LOSS_WEIGHTS = (0.5, 0.5)

# 1. Focal Loss 파라미터
    FOCAL_ALPHA = 0.25         # 클래스 불균형 해소 가중치
    FOCAL_GAMMA = 2.0          # 어려운 샘플에 대한 집중도

    # 2. Jaccard & Dice & Tversky 공용 Smooth
    LOSS_SMOOTH = 1.0

    # 3. Tversky Loss 파라미터 (FP vs FN 가중치)
    TVERSKY_ALPHA = 0.5        # FP(거짓 양성) 벌점
    TVERSKY_BETA = 0.5         # FN(못 찾은 뼈) 벌점 (0.7 등으로 올리면 Recall 상승)

    # 4. Generalized Dice Loss 파라미터
    GDL_SMOOTH = 1e-6
    GDL_GAMMA = 2.0

    # 5. Pixel Weighted BCE 파라미터
    PW_BCE_SMOOTH = 1e-6

    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}