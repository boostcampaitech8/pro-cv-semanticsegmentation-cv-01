import torch.nn as nn
from torchvision import models
from config import Config

def get_model():
    print(">>> [Model] Initializing FCN ResNet50 (Baseline)...")
    
    # 이 파일 안에서 모든 모델 설정을 끝냅니다.
    model = models.segmentation.fcn_resnet50(pretrained=True)
    
    # 출력 채널 수만 Config의 클래스 개수에 맞춤
    model.classifier[4] = nn.Conv2d(512, len(Config.CLASSES), kernel_size=1)
    
    return model