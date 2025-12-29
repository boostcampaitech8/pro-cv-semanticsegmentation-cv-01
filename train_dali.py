import os
import warnings

# ============================================================
# [GPU Selection] torch import 전에 설정!
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

torch.backends.cudnn.benchmark = True

try:
    import wandb
except ImportError:
    wandb = None

from config import Config
from utils import *

def train():
    set_seed(Config.RANDOM_SEED)

    # 1. WandB Init
    if Config.USE_WANDB and wandb is not None:
        run_name = f"{Config.WANDB_RUN_NAME}_DALI"
        if Config.USE_FINETUNE:  # [NEW] Fine-tuning 모드 표시
            run_name += "_finetune"
            
        wandb.init(
            entity=Config.WANDB_ENTITY,
            project=Config.WANDB_PROJECT,
            name=run_name,
            config={
                "epochs": Config.NUM_EPOCHS,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LR,
                "optimizer": Config.OPTIMIZER,
                "loss_function": Config.LOSS_FUNCTION,
                "scheduler": getattr(Config, 'SCHEDULER', 'None'),
                "dataloader": "DALI",
                "dataset_file": Config.DATASET_FILE,
                "finetune": Config.USE_FINETUNE  # [NEW]
            }
        )
        print(f">> WandB Initialized: {Config.WANDB_PROJECT} / {run_name}")

    # 2. Module Load
    try:
        dataset_module = importlib.import_module(Config.DATASET_FILE)
        model_module = importlib.import_module(Config.MODEL_FILE)
    except ModuleNotFoundError as e:
        print(f"Error loading modules: {e}")
        return

    if hasattr(dataset_module, 'get_dali_loader'):
        get_dali_loader = dataset_module.get_dali_loader
    else:
        raise AttributeError(f"Selected dataset file '{Config.DATASET_FILE}' must implement 'get_dali_loader' function.")

    get_model = model_module.get_model

    # ============================================================
    # [MODIFIED] 3. Dataset & Loader - Fine-tuning 지원
    # ============================================================
    print(f">> Initializing DALI DataLoaders from {Config.DATASET_FILE}...")
    train_loader = get_dali_loader(is_train=True, batch_size=Config.BATCH_SIZE)
    
    # Fine-tuning 모드면 validation 없음
    if Config.USE_FINETUNE:
        valid_loader = None
        print(">> Fine-tuning mode: No validation loader")
    else:
        valid_loader = get_dali_loader(is_train=False, batch_size=Config.BATCH_SIZE)
        print(">> Normal mode: Train + Validation loaders ready")
    
    # 4. Model & Optimizer
    model = get_model().cuda()
    
    # Multi-GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f">> Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f">> Using Single GPU")

    # ============================================================
    # [NEW] Pretrained Model Loading (Fine-tuning)
    # ============================================================
    if Config.USE_FINETUNE:
        pretrained_path = os.path.join(Config.SAVED_DIR, "best_model.pt")
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}\n"
                f"Please train a model first with USE_FINETUNE=False"
            )
        
        print(f">> Loading pretrained model from: {pretrained_path}")
        
        # ============================================================
        # [FIX] PyTorch 2.6+ compatibility
        # ============================================================
        saved_model = torch.load(
            pretrained_path, 
            map_location='cuda',
            weights_only=False  # ← 추가!
        )
        
        # 저장된 모델에서 state_dict 추출
        if isinstance(saved_model, nn.DataParallel):
            print("   Detected: Model saved with DataParallel")
            state_dict = saved_model.module.state_dict()
        else:
            print("   Detected: Model saved without DataParallel")
            state_dict = saved_model.state_dict()
        
        # 현재 모델에 로딩
        try:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print(f"✅ Fine-tuning mode activated with {len(train_loader.source)} training images")
        except Exception as e:
            print(f"⚠️ Warning: Could not load state_dict directly: {e}")
            print("   Trying to load with strict=False...")
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded with strict=False")
    
    # Optimizer
    if Config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    elif Config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    elif Config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # Scheduler
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

    # 5. Loss Function
    print(f">> Selected Loss Function: {Config.LOSS_FUNCTION}")
    weights = getattr(Config, 'LOSS_WEIGHTS', (0.5, 0.5))
    
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
        criterion = TverskyLoss(alpha=0.3, beta=0.7)
    elif Config.LOSS_FUNCTION == 'WeightedBCE':
        criterion = PixelWeightedBCE()
    elif Config.LOSS_FUNCTION == 'Combined_WeightedBCE_Dice':
        print(f">> Loss Weights: WeightedBCE({weights[0]}) + Dice({weights[1]})")
        criterion = CombinedLoss(PixelWeightedBCE(), smp_dice, weight_a=weights[0], weight_b=weights[1])
    elif Config.LOSS_FUNCTION == 'Combined_Focal_Dice_Overlap':
        print(f">> Loss: Focal + Dice + Overlap")
        if len(weights) >= 3:
            print(f">> Weights: Focal({weights[0]}) + Dice({weights[1]}) + Overlap({weights[2]})")
        else:
            print(f">> Weights: Focal({weights[0]}) + Dice({weights[1]}) + Overlap(auto)")
        overlap_loss = OverlapPenaltyLoss()
        criterion = CombinedLoss(FocalLoss(), smp_dice, overlap_loss)
    else:
        print("⚠️ Unknown Loss Function. Defaulting to BCE.")
        criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    patience_count = 0

    print(f"\n🚀 Start Training | Opt: {Config.OPTIMIZER} | Scheduler: {scheduler_name}")
    if Config.USE_FINETUNE:
        print(f"🔥 Fine-tuning Mode: Train={len(train_loader.source)} images, No Validation\n")
    else:
        print(f"📊 Normal Mode: Train={len(train_loader.source)} images, Val={len(valid_loader.source)} images\n")

    for epoch in range(Config.NUM_EPOCHS):
        # ==============================
        # Train Loop
        # ==============================
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Warmup
            if getattr(Config, 'USE_WARMUP', False) and epoch < getattr(Config, 'WARMUP_EPOCHS', 5):
                warmup_epochs = Config.WARMUP_EPOCHS
                warmup_min_lr = getattr(Config, 'WARMUP_MIN_LR', 1e-6)
                target_lr = Config.LR
                
                total_warmup_steps = len(train_loader) * warmup_epochs
                current_step = epoch * len(train_loader) + batch_idx
                
                lr = warmup_min_lr + (target_lr - warmup_min_lr) * (current_step / total_warmup_steps)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict): 
                    outputs = outputs["out"]
                
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if Config.USE_WANDB and wandb is not None:
                wandb.log({"train_loss": loss.item()})
        
        gc.collect()
        torch.cuda.empty_cache()
        
        mean_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} | Train Loss: {mean_train_loss:.4f}")

        # ============================================================
        # [MODIFIED] Validation Loop - valid_loader가 있을 때만 실행
        # ============================================================
        if valid_loader is not None and (epoch + 1) % Config.VAL_EVERY == 0:
            model.eval()
            torch.cuda.empty_cache()

            total_val_loss = 0.0
            dices = []

            vis_every = getattr(Config, 'WANDB_VIS_EVERY', 1) 
            do_visualize = (Config.USE_WANDB and wandb is not None and (epoch + 1) % vis_every == 0)
            vis_images, vis_preds, vis_masks = None, None, None

            with torch.no_grad():
                val_pbar = tqdm(valid_loader, desc=f"[Valid] Epoch {epoch+1}")
                for batch_idx, (images, masks) in enumerate(val_pbar):
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(images)
                        if isinstance(outputs, dict): 
                            outputs = outputs["out"]
                        
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                        
                        loss = criterion(outputs, masks)
                        total_val_loss += loss.item()

                    preds = torch.sigmoid(outputs)
                    preds = (preds > 0.5).float()
                    
                    y_pred = preds.flatten(2)
                    y_true = masks.flatten(2)
                    
                    intersection = (y_pred * y_true).sum(dim=2)
                    union = y_pred.sum(dim=2) + y_true.sum(dim=2)
                    
                    batch_dice = (2. * intersection + 1e-4) / (union + 1e-4)
                    dices.append(batch_dice.cpu())

                    if do_visualize and batch_idx == 0:
                        vis_images = images[:1].cpu()
                        vis_preds = preds[:1].cpu()
                        vis_masks = masks[:1].cpu()

            dices = torch.cat(dices, 0)
            dice_per_class = torch.mean(dices, 0)
            val_dice = torch.mean(dice_per_class).item()
            
            mean_val_loss = total_val_loss / len(valid_loader)

            # Scheduler 업데이트
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
            print(f" Mean Dice: {val_dice:.4f}")
            print(f" LR      : {current_lr:.6f}")
            print("-" * 40)
            
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

            # ============================================================
            # [MODIFIED] Save & Early Stopping - Validation 안에서만 실행
            # ============================================================
            min_delta = getattr(Config, 'EARLY_STOPPING_MIN_DELTA', 0.0)
            
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
        
        # ============================================================
        # [NEW] Fine-tuning 모드: 매 epoch마다 모델 저장 (선택사항)
        # ============================================================
        elif Config.USE_FINETUNE:
            # Fine-tuning 시 validation 없으므로 매 epoch 저장 또는 마지막만 저장
            current_lr = optimizer.param_groups[0]['lr']
            print(f">> Fine-tuning Epoch {epoch+1} completed | LR: {current_lr:.6f}")
            
            # Scheduler 업데이트 (ReduceLROnPlateau 제외)
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    # ============================================================
    # [MODIFIED] 최종 모델 저장
    # ============================================================
    if Config.USE_WANDB and wandb is not None:
        wandb.finish()

    # Fine-tuning 모드에서는 마지막 모델 저장
    if Config.USE_FINETUNE:
        save_model(model, Config.SAVED_DIR, "finetuned_model.pt")
        print(f"✅ Fine-tuned model saved to {Config.SAVED_DIR}/finetuned_model.pt")
    elif not Config.SAVE_BEST_MODEL:
        save_model(model, Config.SAVED_DIR, "last_model.pt")

if __name__ == "__main__":
    train()