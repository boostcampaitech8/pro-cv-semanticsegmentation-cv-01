import segmentation_models_pytorch as smp
from config import Config

def get_model():
    # ResNet101: 깊은 층을 사용하여 뼈의 추상적인 특징을 잘 추출합니다.
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(Config.CLASSES)
    )
    return model