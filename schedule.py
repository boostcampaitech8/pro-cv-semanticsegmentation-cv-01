import subprocess
import time

# ====================================================
# 🧪 실험 예약 리스트
# ====================================================
experiments = [
    # 1. Base: BCE + Dice (Standard)
    {
        "exp_name": "WJH_026_hrnet_w18_512_BCE_Dice",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "Combined_BCE_Dice",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
    
    # 2. Hard Mining: Focal + Dice (Ranker Choice)
    {
        "exp_name": "WJH_027_hrnet_w18_512_Focal_Dice",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "Combined_Focal_Dice",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
    
    # 3. Recall Boost: Tversky (For small bone recall)
    {
        "exp_name": "WJH_028_hrnet_w18_512_Tversky",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "Tversky",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
    
    # 4. Imbalance: Generalized Dice
    {
        "exp_name": "WJH_029_hrnet_w18_512_GeneralizedDice",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "GeneralizedDice",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
    
    # 5. Boundary: Pixel Weighted BCE
    {
        "exp_name": "WJH_030_hrnet_w18_512_WeightedBCE",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "WeightedBCE",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
    
    # 6. Pure Dice Loss
    {
        "exp_name": "WJH_031_hrnet_w18_512_Dice",
        "dataset_file": "dataset.dataset_dali_v1",
        "model_file": "model.model_hrnet_w18",
        "loss": "Dice",
        "epoch": 100,
        "resize_size": 512,
        "lr": 5e-5
    },
]

# ====================================================
# 🚀 실행 로직 (자동화)
# ====================================================
for i, exp in enumerate(experiments):
    print(f"\n[Scheduler] {i+1}/{len(experiments)}번째 실험 시작: {exp['exp_name']}")
    
    # 명령어 만들기
    cmd = ["python", "run_exp.py"]
    
    # 딕셔너리에 있는 설정들을 인자로 변환
    for key, value in exp.items():
        cmd.append(f"--{key}")
        
        # 리스트(예: [512, 1024])가 들어오면 풀어서 넣어줌
        if isinstance(value, list) or isinstance(value, tuple):
            for v in value:
                cmd.append(str(v))
        else:
            cmd.append(str(value))
    
    # 실행!
    try:
        subprocess.run(cmd, check=True)
        print(f"[Scheduler] {exp['exp_name']} 완료! 5초 뒤 다음 실험 시작...")
        time.sleep(5) 
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 실험 중 에러 발생: {exp['exp_name']}")
        print("다음 실험으로 넘어갑니다...")
        continue