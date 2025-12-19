import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import GroupKFold

from config import Config

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # 여기에 Flip, Rotate 등 Augmentation 추가 가능
        ])
    else:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        # =============================================================
        # 1. 제외할 ID 목록 정의
        # =============================================================
        exclude_ids = ["ID363", "ID387"]  # 필요한 만큼 추가

# =============================================================
        # 2. 이미지 파일 로드 (수정됨: 전체 경로 검사)
        # =============================================================
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=Config.IMAGE_ROOT)
            for root, _dirs, files in os.walk(Config.IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
            # [핵심 수정] fname(파일명)이 아니라 os.path.join(root, fname)(전체경로)를 검사해야 함
            and not any(ex_id in os.path.join(root, fname) for ex_id in exclude_ids)
        }
        
        # =============================================================
        # 3. 라벨 파일 로드 (수정됨: 전체 경로 검사)
        # =============================================================
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=Config.LABEL_ROOT)
            for root, _dirs, files in os.walk(Config.LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
            # [핵심 수정] 여기도 마찬가지로 전체 경로(root + fname)에서 검사
            and not any(ex_id in os.path.join(root, fname) for ex_id in exclude_ids)
        }
        
        # 디버깅용 출력
        print(f">> [Dataset] {len(exclude_ids)}개의 Artifact ID 제외 완료.")
        print(f"   - Images: {len(pngs)}장")
        print(f"   - Labels: {len(jsons)}장")
        
        # 개수 일치 확인 (안전장치)
        assert len(pngs) == len(jsons), f"이미지와 라벨 개수가 다릅니다! (Img: {len(pngs)}, Lbl: {len(jsons)})"
        
        _filenames = np.array(sorted(pngs))
        _labelnames = np.array(sorted(jsons))
        
        # ... (이하 GroupKFold 로직 동일) ...
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
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(Config.IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
            
        label_name = self.labelnames[item]
        label_path = os.path.join(Config.LABEL_ROOT, label_name)
        
        label_shape = tuple(image.shape[:2]) + (len(Config.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = Config.CLASS2IND[c]
            points = np.array(ann["points"], dtype=np.int32)
            
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        # [수정된 핵심 부분]
        # Train/Valid 여부와 상관없이 항상 이미지와 마스크를 함께 Transform에 넘깁니다.
        # 이렇게 해야 Valid 때도 마스크가 512x512로 리사이즈됩니다.
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"]

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
    
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

        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), image_name