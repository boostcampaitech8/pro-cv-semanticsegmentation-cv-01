import os
import cv2
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from config import Config

# =============================================================================
# 🔧 전처리 설정 (A.OneOf로 크기 에러 원천 차단)
# =============================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            # 🔥 [핵심] 100% 확률(p=1.0)로 둘 중 하나를 실행 -> 결과물은 무조건 512x512
            A.OneOf([
                # 옵션 1: 줌인 (확대 학습) - 디테일
                A.RandomResizedCrop(
                    height=Config.RESIZE_SIZE[0], 
                    width=Config.RESIZE_SIZE[1], 
                    scale=(0.5, 1.0), 
                    ratio=(0.75, 1.33), 
                    p=1.0 
                ),
                # 옵션 2: 전체 보기 - 문맥
                A.Resize(
                    height=Config.RESIZE_SIZE[0], 
                    width=Config.RESIZE_SIZE[1],
                    p=1.0
                )
            ], p=1.0), 
            
            # 선명도 강화 (Train/Valid 모두 적용)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            # 🔥 [중요] 여기서 Tensor로 자동 변환됨 (HWC -> CHW 자동 처리)
            ToTensorV2()
        ])
    else:
        # 검증용
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# =============================================================================
# 💿 Dataset 클래스 (OpenCV 에러 및 Transpose 충돌 해결)
# =============================================================================
class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        self.is_train = is_train
        self.transforms = transforms
        
        # 사용자님의 원본 로딩 방식 유지 (경로 호환성 확보)
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=Config.IMAGE_ROOT)
            for root, _dirs, files in os.walk(Config.IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=Config.LABEL_ROOT)
            for root, _dirs, files in os.walk(Config.LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        _filenames = np.array(sorted(pngs))
        _labelnames = np.array(sorted(jsons))
        
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for fname in _filenames]
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == 0: continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break
        
        self.filenames = filenames
        self.labelnames = labelnames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(Config.IMAGE_ROOT, image_name)
        
        # 1. 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"{image_path} Not Found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 라벨 로드
        label_name = self.labelnames[item]
        label_path = os.path.join(Config.LABEL_ROOT, label_name)
        
        label_shape = tuple(image.shape[:2]) + (len(Config.CLASSES), )
        mask = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = Config.CLASS2IND[c]
            points = np.array(ann["points"], dtype=np.int32)
            
            # 🔥 [Fix 1] OpenCV fillPoly 메모리 에러 방지 (임시 배열 사용)
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            mask[..., class_ind] = class_label

        # 3. Transform 적용
        if self.transforms:
            inputs = {"image": image, "mask": mask}
            result = self.transforms(**inputs)
            image = result["image"] # 얘는 ToTensorV2 덕분에 이미 (3, 512, 512)
            mask = result["mask"]   # 얘는 아직 (512, 512, 29) 상태임!

        # 🔥 [수정] 마스크를 (H, W, C) -> (C, H, W)로 바꿔줘야 함
        # mask가 텐서라면 .permute, numpy라면 .transpose를 써야 하는데
        # ToTensorV2를 거쳤으면 텐서일 확률이 높지만, 안전하게 처리합니다.
        
        if isinstance(mask, torch.Tensor):
            mask = mask.permute(2, 0, 1) # (512, 512, 29) -> (29, 512, 512)
        else:
            # 만약 텐서가 아니라면 (혹시 모를 대비)
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask
        
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        self.image_root = Config.TEST_IMAGE_ROOT
        self.filenames = np.array(sorted([
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ]))
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result["image"]

        #     이미 Tensor이고 CHW 형태임
        # 하지만 transforms 없거나 ToTensorV2가 없는 경우를 대비
        if isinstance(image, np.ndarray):
            image = image.transpose(2, 0, 1) # HWC -> CHW
            return torch.from_numpy(image).float(), image_name
            
        # Tensor인 경우 (ToTensorV2 적용됨)
        return image.float(), image_name
