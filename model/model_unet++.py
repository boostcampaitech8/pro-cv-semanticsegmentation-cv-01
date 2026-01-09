import segmentation_models_pytorch as smp
from config import Config

def get_model():
    # EfficientNet-b4: B0보다 파라미터가 많아 더 복잡한 특징을 잘 잡습니다.
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", 
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(Config.CLASSES)
    )
    return model