from config import Config

# 필요한 라이브러리는 이 파일에서만 import
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

def get_model():
    if smp is None:
        raise ImportError("Install segmentation_models_pytorch first!")

    print(">>> [Model] Initializing SMP Unet with EfficientNet-B0...")

    # 내가 원하는 설정을 여기에 직접 하드코딩
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # 여기를 resnet101로 바꾸면 다른 모델이 됨
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(Config.CLASSES)
    )
    
    return model