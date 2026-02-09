
import os
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import GroupKFold
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from config import Config

# ============================================================
# [슬라이딩 윈도우 버전]
# 1024x1024 윈도우 크기, 스트라이드 1024 (2x2 패치 구성)
# ============================================================

def get_transforms(is_train=True):
    """
    Sliding Window에서는 Resize 제거!
    원본 2048 크기 유지 후 GPU에서 슬라이딩 윈도우 크롭
    """
    if is_train:
        return A.Compose([
            # 리사이즈 제거: 원본 해상도(2048)를 그대로 유지합니다.
            # A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            
            # CLAHE 적용 (CPU 기반)
            A.CLAHE(clip_limit=Config.CLAHE_CLIP_LIMIT, tile_grid_size=Config.CLAHE_TILE_GRID_SIZE, p=1.0),
            
            # SSR(Shift, Scale, Rotate) 등은 DALI를 통해 GPU에서 처리됩니다.
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05,
                rotate_limit=20,     
                p=0.5,              
                border_mode=0       
            ),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            # 리사이즈 제거!
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# ============================================================
# [로직] get_transforms 파싱
# ============================================================
class DaliTransformParser:
    def __init__(self, is_train=True):
        self.is_train = is_train
        transforms = get_transforms(is_train)
        
        # 슬라이딩 윈도우 설정: Config에서 윈도우 크기와 스트라이드 값을 가져옵니다.
        self.window_size = getattr(Config, 'WINDOW_SIZE', 1024)
        self.stride = getattr(Config, 'STRIDE', 1024)
        
        # 기본값 설정
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.cpu_transforms = []
        
        self.use_geometric = False
        self.geo_prob = 0.0
        self.rotate_limit = 0; self.scale_limit = 0; self.shift_limit = 0
        self.use_hflip = False; self.hflip_prob = 0.0

        if transforms:
            t_list = transforms.transforms if isinstance(transforms, A.Compose) else transforms
            for t in t_list:
                if isinstance(t, A.Resize):
                    # Resize는 무시 (원본 크기 유지)
                    pass
                    
                elif isinstance(t, A.Normalize):
                    self.mean = [m * 255 for m in t.mean]
                    self.std = [s * 255 for s in t.std]
                    
                elif isinstance(t, A.ShiftScaleRotate):
                    self.use_geometric = True
                    self.geo_prob = t.p
                    
                    r_lim = max(abs(t.rotate_limit[0]), abs(t.rotate_limit[1]))
                    s_lim = max(abs(t.scale_limit[0]), abs(t.scale_limit[1]))
                    sx = max(abs(t.shift_limit_x[0]), abs(t.shift_limit_x[1]))
                    sy = max(abs(t.shift_limit_y[0]), abs(t.shift_limit_y[1]))
                    sh_lim = max(sx, sy)
                    
                    self.rotate_limit = r_lim
                    self.scale_limit = s_lim
                    self.shift_limit = sh_lim 
                    
                elif isinstance(t, A.HorizontalFlip):
                    self.use_hflip = True
                    self.hflip_prob = t.p
                elif isinstance(t, (A.Compose, A.OneOf)):
                    pass 
                else:
                    # 예: CLAHE
                    self.cpu_transforms.append(t)
        
        self.cpu_compose = A.Compose(self.cpu_transforms) if self.cpu_transforms else None
        
    def get_affine_matrix(self, h, w):
        matrix = np.eye(3, dtype=np.float32)[:2, :] 
        
        if self.is_train and self.use_geometric and random.random() < self.geo_prob:
            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            shift_x = random.uniform(-self.shift_limit, self.shift_limit)
            shift_y = random.uniform(-self.shift_limit, self.shift_limit)
            
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += shift_x * w
            M[1, 2] += shift_y * h
            matrix = cv2.invertAffineTransform(M).astype(np.float32)
            
        return matrix

# ============================================================
# 외부 소스 (슬라이딩 윈도우 방식)
# ============================================================
class XRayExternalSource:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.parser = DaliTransformParser(is_train)
        
        # JPEG Root 경로 설정
        self.jpeg_root = Config.IMAGE_ROOT.rstrip('/') + "_jpeg"
        if not os.path.exists(self.jpeg_root):
             raise FileNotFoundError(f"JPEG Root not found: {self.jpeg_root}. Run tools/preprocess_to_jpeg.py first.")
            
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
        val_fold = getattr(Config, 'VAL_FOLD', 4)

        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if (i == val_fold) ^ is_train:
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
        
        self.filenames = list(filenames)
        self.labelnames = list(labelnames)
        
        # 슬라이딩 윈도우: 각 원본 이미지를 4개의 패치로 분할하여 확장합니다 (2x2 구성).
        # 스트라이드 1024 기준: (2048-1024)//1024 + 1 = 2개 → 가로세로 조합 시 총 4개 패치 생성
        self.window_size = self.parser.window_size
        self.stride = self.parser.stride
        self.patches_per_image = ((2048 - self.window_size) // self.stride + 1) ** 2
        
        self.n = len(self.filenames) * self.patches_per_image
        
        print(f"[Sliding Window] Images: {len(self.filenames)}, Patches per image: {self.patches_per_image}, Total patches: {self.n}")
        
        self.indices = list(range(len(self.filenames)))
        if is_train:
            random.shuffle(self.indices)
            
        # [캐시] 이미지와 라벨을 캐시하여 중복 로딩을 방지합니다.
        self.last_image_idx = -1
        self.cached_image = None
        self.cached_label = None

    def __len__(self):
        return self.n

    def __call__(self, sample_info):
        idx_in_epoch = sample_info.idx_in_epoch
        if idx_in_epoch >= self.n: 
            idx_in_epoch %= self.n
        
        # 현재 인덱스가 어떤 이미지의 몇 번째 패치에 해당하는지 계산합니다.
        image_idx = idx_in_epoch // self.patches_per_image
        patch_idx = idx_in_epoch % self.patches_per_image
        
        real_idx = self.indices[image_idx]
        
        # [캐시 확인] 이전에 로드된 이미지와 동일한 경우 캐시된 데이터를 사용합니다.
        if self.last_image_idx == real_idx and self.cached_image is not None:
            image = self.cached_image
            label = self.cached_label
        else:
            image_name = self.filenames[real_idx]
            
            # JPEG 파일 경로
            jpeg_name = os.path.splitext(image_name)[0] + ".jpg"
            image_path = os.path.join(self.jpeg_root, jpeg_name)
            
            # 1. JPEG 이미지 로드 (원본 2048x2048 해상도)
            image = cv2.imread(image_path)
            if image is None: 
                raise FileNotFoundError(f"JPEG not found: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. CLAHE 적용 (2048 크기 이미지에 적용)
            if self.parser.cpu_compose:
                image = self.parser.cpu_compose(image=image)["image"]
            
            # 3. 마스크 생성 (2048x2048)
            label_name = self.labelnames[real_idx]
            label_path = os.path.join(Config.LABEL_ROOT, label_name)
            label_shape = (2048, 2048, len(Config.CLASSES))
            label = np.zeros(label_shape, dtype=np.uint8)
            
            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]
            
            for ann in annotations:
                c = ann["label"]
                class_ind = Config.CLASS2IND[c]
                points = np.array(ann["points"], dtype=np.int32)
                
                class_label = np.zeros((2048, 2048), dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label
            
            # Update Cache
            self.last_image_idx = real_idx
            self.cached_image = image
            self.cached_label = label
        
        # 4. 패치 좌표 계산 및 CPU에서 크롭 (고정 그리드)
        patches_per_row = (2048 - self.window_size) // self.stride + 1
        row = patch_idx // patches_per_row
        col = patch_idx % patches_per_row
        crop_y = row * self.stride
        crop_x = col * self.stride
        
        # CPU에서 크롭 (1024x1024)
        image_patch = image[crop_y:crop_y+self.window_size, crop_x:crop_x+self.window_size]
        label_patch = label[crop_y:crop_y+self.window_size, crop_x:crop_x+self.window_size]
        
        # 5. Affine 행렬 (윈도우 크기 기준)
        matrix = self.parser.get_affine_matrix(self.window_size, self.window_size)

        # Return: Image(1024x1024), Mask(1024x1024), Matrix
        return image_patch, label_patch, matrix
    
    def shuffle(self):
        if self.is_train:
            random.shuffle(self.indices)

# ============================================================
# DALI Pipeline (Sliding Window)
# ============================================================
class XRayDaliPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_source, py_num_workers=1):
        super(XRayDaliPipeline, self).__init__(
            batch_size, 
            num_threads, 
            device_id, 
            seed=12, 
            py_start_method="spawn",
            py_num_workers=py_num_workers
        )
        self.source = external_source
        self.parser = external_source.parser 
        self.window_size = self.parser.window_size
        
    def define_graph(self):
        # inputs: 0=Image(1024x1024), 1=Mask(1024x1024), 2=Matrix
        self.images, self.masks, self.matrices = fn.external_source(
            source=self.source, 
            num_outputs=3, 
            dtype=[types.UINT8, types.UINT8, types.FLOAT],
            parallel=True,       
            batch=False,         
            prefetch_queue_depth=8 
        )
        
        # Transfer to GPU
        images = self.images.gpu() 
        masks = self.masks.gpu() 
        matrices = self.matrices 
        
        # GPU Affine (윈도우 크기에 적용)
        if self.parser.use_geometric:
             images = fn.warp_affine(images, matrix=matrices, interp_type=types.INTERP_LINEAR, fill_value=0)
             masks = fn.warp_affine(masks, matrix=matrices, interp_type=types.INTERP_NN, fill_value=0)
        
        # HFlip
        mirror = None
        if self.parser.use_hflip:
             mirror = fn.random.coin_flip(probability=self.parser.hflip_prob)

        # Normalization
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mirror=mirror,
            mean=self.parser.mean, 
            std=self.parser.std    
        )
        
        if mirror is not None:
             masks = fn.flip(masks, horizontal=mirror)
             
        masks = fn.transpose(masks, perm=[2, 0, 1])
        masks = fn.cast(masks, dtype=types.FLOAT)
        
        return images, masks

# ============================================================
# Loader Wrapper
# ============================================================
class DaliDataLoaderWrapper:
    def __init__(self, pipeline, source, batch_size):
        self.pipeline = pipeline
        self.source = source
        self.batch_size = batch_size
        self.iterator = DALIGenericIterator(
            pipelines=[pipeline], 
            output_map=["image", "mask"], 
            size=len(source),
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )
        
    def __len__(self):
        return (len(self.source) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = next(self.iterator)
            batch = data[0]
            return batch["image"], batch["mask"]
        except StopIteration:
            self.source.shuffle()
            raise StopIteration

def get_dali_loader(is_train=True, batch_size=None):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
        
    e_source = XRayExternalSource(is_train=is_train)
    
    num_workers = getattr(Config, 'NUM_WORKERS', 8)
    dali_threads = getattr(Config, 'DALI_NUM_THREADS', 4) 
    device_id = getattr(Config, 'DEVICE_ID', 0)
    
    pipe = XRayDaliPipeline(
        batch_size=batch_size, 
        num_threads=dali_threads, 
        device_id=device_id, 
        external_source=e_source,
        py_num_workers=num_workers
    )
    pipe.build()
    
    return DaliDataLoaderWrapper(pipe, source=e_source, batch_size=batch_size)

# ============================================================
# Inference Dataset (Sliding Window)
# ============================================================
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
        
        # Sliding Window 설정
        self.window_size = getattr(Config, 'WINDOW_SIZE', 1024)
        self.stride = getattr(Config, 'STRIDE', 1024)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        슬라이딩 윈도우용: 원본 2048 이미지 반환
        패치 분할은 inference 스크립트에서 처리
        """
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 원본 2048 크기 유지, transform 적용 (CLAHE 등)
        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result["image"]

        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float(), image_name
