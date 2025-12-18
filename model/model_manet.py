import segmentation_models_pytorch as smp
from config import Config

def get_model():
    # se_resnext50_32x4d: 강력한 ResNeXt 인코더에 SE(Squeeze-Excitation) Attention 추가
    model = smp.MAnet(
        encoder_name="se_resnext50_32x4d",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(Config.CLASSES)
    )
    return model