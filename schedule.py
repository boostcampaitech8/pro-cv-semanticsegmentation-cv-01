import subprocess
import time

# ====================================================
# 🧪 실험 예약 리스트 (원하는 조합을 여기에 적으세요)
# ====================================================
experiments = [
    # 실험 1: 전처리 비교 (CLAHE 적용)
    {
        "exp_name": "Exp01_CLAHE_Base",
        "dataset_file": "dataset.dataset_clahe",
        "model_file": "model.model_unet",
        "loss": "Combined_BCE_Dice",
        "epoch": 25
    },
    # 실험 2: 모델 변경 (SegFormer)
    {
        "exp_name": "Exp02_SegFormer_Focal",
        "dataset_file": "dataset.dataset_clahe",
        "model_file": "model.model_segformer",
        "loss": "Combined_Focal_Dice",
        "epoch": 50
    },
    # 실험 3: LR 변경 테스트
    {
        "exp_name": "Exp03_UNet_LowLR",
        "dataset_file": "dataset.dataset_clahe",
        "model_file": "model.model_unet",
        "lr": 1e-5,
        "epoch": 25
    },
]

# ====================================================
# 🚀 실행 로직 (자동화)
# ====================================================
for i, exp in enumerate(experiments):
    print(f"\n[Scheduler] {i+1}/{len(experiments)}번째 실험 시작: {exp['exp_name']}")
    
    # 명령어 만들기
    cmd = ["python", "run_exp.py"]
    
    # 딕셔너리에 있는 설정들을 인자로 변환 (--key value)
    for key, value in exp.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    
    # 실행! (subprocess가 프로세스를 새로 띄워서 실행함 -> 메모리 초기화됨)
    try:
        subprocess.run(cmd, check=True)
        print(f"[Scheduler] {exp['exp_name']} 완료! 5초 뒤 다음 실험 시작...")
        time.sleep(5) # 잠시 휴식 (GPU 열 식히기 + WandB 동기화 시간 벌기)
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 실험 중 에러 발생: {exp['exp_name']}")
        print("다음 실험으로 넘어갑니다...")
        continue