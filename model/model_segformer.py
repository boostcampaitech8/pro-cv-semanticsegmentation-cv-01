import segmentation_models_pytorch as smp
from config import Config

def get_model():
    # mit_b2: SegFormer 전용 인코더 (b0~b5 중 b2가 가성비 좋음)
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(Config.CLASSES)
    )
        # --- 여기에 추가하세요 ---
    if hasattr(model.encoder, 'set_grad_checkpointing'):
        model.encoder.set_grad_checkpointing(True)
        print(">> Gradient Checkpointing Enabled.")
    # -----------------------
    
    return model