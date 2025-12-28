import os
import warnings

# ============================================================
# [GPU Selection] torch import 전에 설정!
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # GPU 0,1 사용 (필요시 수정)

# [Warning Suppression]
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", message="Please set `reader_name`")

import importlib
import numpy as np
import gc
import segmentation_models_pytorch as smp 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

# 벤치마크 모드
torch.backends.cudnn.benchmark = True

try:
    import wandb
except ImportError:
    wandb = None

from config import Config
from utils import *

# DALI Loader
# from dataset.dataset_dali import get_dali_loader  <-- Removed static import

def train():
    set_seed(Config.RANDOM_SEED)

    # 1. WandB Init
    if Config.USE_WANDB and wandb is not None:
        wandb.init(
            entity=Config.WANDB_ENTITY,
            project=Config.WANDB_PROJECT,
            name=f"{Config.WANDB_RUN_NAME}_DALI", # Suffix added
            config={
                "epochs": Config.NUM_EPOCHS,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LR,
                "optimizer": Config.OPTIMIZER,
                "loss_function": Config.LOSS_FUNCTION,
                "scheduler": getattr(Config, 'SCHEDULER', 'None'),
                "dataloader": "DALI",
                "dataset_file": Config.DATASET_FILE
            }
        )
        print(f">> WandB Initialized: {Config.WANDB_PROJECT} / {Config.WANDB_RUN_NAME}_DALI")

    # 2. Module Load
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE) # [Modified] Dynamic Import
        model_module = importlib.import_module(Config.MODEL_FILE)
    except ModuleNotFoundError as e:
        print(f"Error loading modules: {e}")
        return

    # Check for DALI loader function
    if hasattr(dataset_module, 'get_dali_loader'):
        get_dali_loader = dataset_module.get_dali_loader
    else:
        raise AttributeError(f"Selected dataset file '{Config.DATASET_FILE}' must implement 'get_dali_loader' function.")

    get_model = model_module.get_model

    # 3. Dataset & Loader (REPLACED WITH DALI)
    print(f">> Initializing DALI DataLoaders from {Config.DATASET_FILE}...")
    # 3. Dataset & Loader (REPLACED WITH DALI)
    print(f">> Initializing DALI DataLoaders from {Config.DATASET_FILE}...")
    train_loader = get_dali_loader(is_train=True, batch_size=Config.BATCH_SIZE)
    valid_loader = get_dali_loader(is_train=False, batch_size=Config.BATCH_SIZE)
    print(">> DALI DataLoaders Ready.")
    
    # 4. Model & Optimizer
    model = get_model().cuda()
    # ============================================================
    # [Multi-GPU] DataParallel Support
    # ============================================================
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f">> Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f">> Using Single GPU")



    if Config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    elif Config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    elif Config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # Config 기반 스케줄러 선택
    scheduler_name = getattr(Config, 'SCHEDULER', 'ReduceLROnPlateau')
    scheduler = None
    
    print(f">> Selected Scheduler: {scheduler_name}")

    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=Config.SCHEDULER_FACTOR, 
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=getattr(Config, 'SCHEDULER_MIN_LR', 1e-6),
            verbose=True
        )
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=Config.SCHEDULER_STEP_SIZE, 
            gamma=Config.SCHEDULER_GAMMA
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.SCHEDULER_T_MAX, 
            eta_min=getattr(Config, 'SCHEDULER_MIN_LR', 1e-6)
        )

    scaler = torch.amp.GradScaler()

    # 5. Loss Function Selector
    print(f">> Selected Loss Function: {Config.LOSS_FUNCTION}")
    weights = getattr(Config, 'LOSS_WEIGHTS', (0.5, 0.5))
    
    # [공통] SMP Dice Loss 생성 (재사용)
    smp_dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True, smooth=Config.LOSS_SMOOTH)

    if Config.LOSS_FUNCTION == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        
    elif Config.LOSS_FUNCTION == 'Focal':
        criterion = FocalLoss(alpha=0.25, gamma=2)
        
    elif Config.LOSS_FUNCTION == 'Dice':
        criterion = smp_dice
        
    elif Config.LOSS_FUNCTION == 'Combined_BCE_Dice':
        criterion = CombinedLoss(nn.BCEWithLogitsLoss(), smp_dice, weights[0], weights[1])
        
    elif Config.LOSS_FUNCTION == 'Combined_Focal_Dice':
        criterion = CombinedLoss(FocalLoss(), smp_dice, weights[0], weights[1])
        
    elif Config.LOSS_FUNCTION == 'GeneralizedDice':
        criterion = GeneralizedDiceLoss()
        
    elif Config.LOSS_FUNCTION == 'Combined_GDL_BCE':
        criterion = CombinedLoss(GeneralizedDiceLoss(), nn.BCEWithLogitsLoss(), weights[0], weights[1])

    elif Config.LOSS_FUNCTION == 'Tversky':
        criterion = TverskyLoss(alpha=0.3, beta=0.7) # Alpha(FP), Beta(FN) -> Recall Emphasis

    elif Config.LOSS_FUNCTION == 'WeightedBCE':
        criterion = PixelWeightedBCE()

    elif Config.LOSS_FUNCTION == 'Combined_WeightedBCE_Dice':
        print(f">> Loss Weights: WeightedBCE({weights[0]}) + Dice({weights[1]})")
        criterion = CombinedLoss(
            PixelWeightedBCE(), 
            smp_dice, 
            weight_a=weights[0], 
            weight_b=weights[1]
        )
        
    else:
        print("⚠️ Unknown Loss Function. Defaulting to BCE.")
        criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    patience_count = 0

    print(f"\n🚀 Start Training | Opt: {Config.OPTIMIZER} | Scheduler: {scheduler_name}\n")

    for epoch in range(Config.NUM_EPOCHS):
        # ==============================
        # Train Loop
        # ==============================
        
        # [Restored] Standard DALI Usage (No Re-init)
        # We rely on the initial shuffle. This matches the fast 'kiwi5215' run.
        
        model.train()
        train_losses = []

        # DALI Loader Integration:
        # DALI loader is iterable, typically wrapped to return (image, mask).
        # We need to manually handle tqdm if len(loader) works.
        
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # images, masks are already on GPU usually if DALI Pipeline outputs GPU tensors.
            # But the wrapper returns PyTorch tensors.
            # DALIWrapper output: image, mask (from our next() implementation).
            # Check dataset_dali.py:
            # images = self.jpegs.gpu() <-- .gpu()
            # so they are CUDA tensors.
            
            # images = images.cuda(non_blocking=True) # Redundant if already on GPU but safe
            # masks = masks.cuda(non_blocking=True)
            
            # Ensure they are correct type (Float)
            # DALI types.FLOAT used in pipeline, so good.
            
            # [NEW] Manual Warmup Logic
            if getattr(Config, 'USE_WARMUP', False) and epoch < getattr(Config, 'WARMUP_EPOCHS', 5):
                warmup_epochs = Config.WARMUP_EPOCHS
                warmup_min_lr = getattr(Config, 'WARMUP_MIN_LR', 1e-6)
                target_lr = Config.LR
                
                # Global Step Calculation
                total_warmup_steps = len(train_loader) * warmup_epochs
                current_step = epoch * len(train_loader) + batch_idx  # batch_idx needed
                
                # Linear Interpolation
                lr = warmup_min_lr + (target_lr - warmup_min_lr) * (current_step / total_warmup_steps)
                
                # Apply LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs["out"]
                
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            
            # [Fix] Gradient Clipping for AMP Stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if Config.USE_WANDB and wandb is not None:
                wandb.log({"train_loss": loss.item()})
        
        # Epoch Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        mean_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} | Train Loss: {mean_train_loss:.4f}")

        # ==============================
        # Validation Loop (Mean Dice 방식 적용)
        # ==============================
        if (epoch + 1) % Config.VAL_EVERY != 0:
            continue

        model.eval()
        torch.cuda.empty_cache()

        total_val_loss = 0.0
        dices = []  # [NEW] 배치별 Dice 점수를 모을 리스트

        # 시각화 설정
        vis_every = getattr(Config, 'WANDB_VIS_EVERY', 1) 
        do_visualize = (Config.USE_WANDB and wandb is not None and (epoch + 1) % vis_every == 0)
        vis_images, vis_preds, vis_masks = None, None, None

        with torch.no_grad():
            val_pbar = tqdm(valid_loader, desc=f"[Valid] Epoch {epoch+1}")
            for batch_idx, (images, masks) in enumerate(val_pbar):
                # images = images.cuda(non_blocking=True)
                # masks = masks.cuda(non_blocking=True)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    if isinstance(outputs, dict): outputs = outputs["out"]
                    
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    
                    # Loss 계산
                    loss = criterion(outputs, masks)
                    total_val_loss += loss.item()

                # [NEW] Prediction 후처리 (Sigmoid -> Threshold)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                
                # [NEW] 배치 내 각 이미지별 Dice 계산 (Mean Dice Logic)
                # (Batch, Class, H, W) -> (Batch, Class, -1)
                y_pred = preds.flatten(2)
                y_true = masks.flatten(2)
                
                # Intersection & Union per sample
                intersection = (y_pred * y_true).sum(dim=2) # (B, C)
                union = y_pred.sum(dim=2) + y_true.sum(dim=2) # (B, C)
                
                # Dice per sample in batch
                batch_dice = (2. * intersection + 1e-4) / (union + 1e-4) # (B, C)
                dices.append(batch_dice.cpu()) # CPU로 옮겨서 저장

                # 시각화 데이터 저장 (첫 배치만)
                if do_visualize and batch_idx == 0:
                    vis_images = images[:1].cpu()
                    vis_preds = preds[:1].cpu()
                    vis_masks = masks[:1].cpu()

        # [NEW] 전체 평균 Dice 계산
        dices = torch.cat(dices, 0) # (Total_Images, Num_Classes)
        dice_per_class = torch.mean(dices, 0) # 클래스별 평균 (C,)
        val_dice = torch.mean(dice_per_class).item() # 전체 평균 (Scalar)
        
        mean_val_loss = total_val_loss / len(valid_loader)

        # 스케줄러 업데이트
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_val_loss)
            else:
                scheduler.step()

        # 결과 출력
        current_lr = optimizer.param_groups[0]['lr']
        print("\n" + "=" * 40)
        print(f" Epoch {epoch+1} Validation Summary")
        print("-" * 40)
        print(f" Val Loss: {mean_val_loss:.4f}")
        print(f" Mean Dice: {val_dice:.4f}") # Mean Dice 출력
        print(f" LR      : {current_lr:.6f}")
        print("-" * 40)
        
        # 클래스별 점수 출력
        for i, class_name in enumerate(Config.CLASSES):
            print(f" {class_name:<15}: {dice_per_class[i].item():.4f}")
        print("=" * 40 + "\n")

        # WandB Log
        if Config.USE_WANDB and wandb is not None:
            log_dict = {
                "val_dice": val_dice,
                "val_loss": mean_val_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1
            }

            if do_visualize and vis_images is not None:
                img = vis_images[0].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img = (img * 255).astype(np.uint8)
                
                pred = vis_preds[0][0].numpy()
                pred = (pred * 255).astype(np.uint8)
                pred = np.stack([pred]*3, axis=-1)
                
                gt = vis_masks[0][0].numpy()
                gt = (gt * 255).astype(np.uint8)
                gt = np.stack([gt]*3, axis=-1)

                log_dict["Visualization"] = [
                    wandb.Image(img, caption="Input"),
                    wandb.Image(pred, caption="Prediction"),
                    wandb.Image(gt, caption="GT"),
                ]
            wandb.log(log_dict)

        # Save & Early Stopping
        # Config에서 값 가져오기 (없으면 기본값 0.0)
        min_delta = getattr(Config, 'EARLY_STOPPING_MIN_DELTA', 0.0)
        
        # [핵심 수정] (기존 점수 + 최소 변화량)보다 커야 갱신으로 인정!
        if val_dice > (best_dice + min_delta):
            print(f"🔥 Best Dice Updated: {best_dice:.4f} → {val_dice:.4f} (Delta: {val_dice - best_dice:.4f})")
            best_dice = val_dice
            patience_count = 0
            if Config.SAVE_BEST_MODEL:
                save_model(model, Config.SAVED_DIR, "best_model.pt")
        else:
            patience_count += 1
            print(f"No improvement. Patience {patience_count}/{Config.EARLY_STOPPING_PATIENCE}")

        if Config.USE_EARLY_STOPPING and patience_count >= Config.EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping triggered.")
            break

    if Config.USE_WANDB and wandb is not None:
        wandb.finish()

    if not Config.SAVE_BEST_MODEL:
        save_model(model, Config.SAVED_DIR, "last_model.pt")

if __name__ == "__main__":
    train()
