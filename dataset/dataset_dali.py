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

# [USER CONFIG AREA] match dataset_final.py style!
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05,
                rotate_limit=20,     
                p=0.5,              
                border_mode=0       
            ),
            # A.HorizontalFlip(p=0.5), # Uncomment to enable
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# [LOGIC] Parse A.Compose to separate DALI (GPU) vs CPU ops
class DaliTransformParser:
    def __init__(self, transforms, is_train=True):
        self.is_train = is_train
        
        # Defaults
        self.resize_shape = Config.RESIZE_SIZE # Default from Config
        
        self.use_geometric = False
        self.geo_prob = 0.5
        self.rotate_limit = 0
        self.scale_limit = 0
        self.shift_limit = 0
        
        self.use_hflip = False
        self.hflip_prob = 0.5
        
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        
        self.cpu_transforms = []
        
        if transforms:
            for t in transforms:
                if isinstance(t, A.Resize):
                    self.resize_shape = (t.height, t.width)
                    # Do not add Resize to cpu_transforms (handled manually)
                elif isinstance(t, A.ShiftScaleRotate):
                    self.use_geometric = True
                    self.geo_prob = t.p
                    # Access tuple limits (min, max). We usually want the max extent for limit.
                    # shift_limit is usually symmetric if passed as float.
                    # A.ShiftScaleRotate stores it as shift_limit_x, shift_limit_y tuples.
                    self.rotate_limit = max(abs(t.rotate_limit[0]), abs(t.rotate_limit[1]))
                    self.scale_limit = max(abs(t.scale_limit[0]), abs(t.scale_limit[1]))
                    # Take max of x/y shift
                    sx = max(abs(t.shift_limit_x[0]), abs(t.shift_limit_x[1]))
                    sy = max(abs(t.shift_limit_y[0]), abs(t.shift_limit_y[1]))
                    self.shift_limit = max(sx, sy)
                elif isinstance(t, A.HorizontalFlip):
                    self.use_hflip = True
                    self.hflip_prob = t.p
                elif isinstance(t, A.Normalize):
                    self.mean = [m * 255 for m in t.mean]
                    self.std = [s * 255 for s in t.std]
                elif isinstance(t, (A.Compose, A.OneOf)):
                     pass
                else:
                    # Treat as CPU transform (e.g., CLAHE, Crop, Noise)
                    self.cpu_transforms.append(t)
        
        # Create separate compose for CPU parts
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

class XRayExternalSource:
    def __init__(self, is_train=True):
        self.is_train = is_train
        
        # 1. Get User Transforms
        transforms = get_transforms(is_train)
        
        # 2. Parse them for DALI
        self.parser = DaliTransformParser(transforms, is_train)
            
        # --- Data Loading Logic ---
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
                if i == 4: continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                if i == 4:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.n = len(self.filenames)
        
        if self.is_train:
            combined = list(zip(self.filenames, self.labelnames))
            random.shuffle(combined)
            self.filenames, self.labelnames = zip(*combined)
            self.filenames = list(self.filenames)
            self.labelnames = list(self.labelnames)

    def __call__(self, sample_info):
        idx = sample_info.idx_in_epoch
        if idx >= self.n:
            idx = idx % self.n
            
        image_name = self.filenames[idx]
        image_path = os.path.join(Config.IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        h_orig, w_orig = image.shape[:2]
        h_target, w_target = self.parser.resize_shape # Used parsed shape
        
        # [OPTIMIZATION] Immediate Resize
        if (h_orig, w_orig) != (h_target, w_target):
            image = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        
        # [OPTIMIZATION] Efficient Mask Gen
        label_name = self.labelnames[idx]
        label_path = os.path.join(Config.LABEL_ROOT, label_name)
        
        label_shape = (h_target, w_target, len(Config.CLASSES))
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        scale_x = w_target / w_orig
        scale_y = h_target / h_orig
        
        for ann in annotations:
            c = ann["label"]
            class_ind = Config.CLASS2IND[c]
            points = np.array(ann["points"], dtype=np.float32)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points = points.astype(np.int32)
            
            class_label = np.zeros((h_target, w_target), dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
            
        # [CPU Transform] Extracted from A.Compose (e.g. CLAHE, Crop)
        if self.parser.cpu_compose:
            # Pass BOTH image and mask so spatial transforms (Crop) work on both
            inputs = {"image": image, "mask": label}
            result = self.parser.cpu_compose(**inputs)
            image = result["image"]
            label = result["mask"]
        
        # [Help] Get Matrix for GPU Geometric Augmentation
        matrix = self.parser.get_affine_matrix(h_target, w_target)

        return image, label, matrix

    def __len__(self):
        return self.n

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
        self.parser = external_source.parser # Share parsed config
        
    def define_graph(self):
        self.images, self.masks, self.matrices = fn.external_source(
            source=self.source, 
            num_outputs=3, 
            dtype=[types.UINT8, types.UINT8, types.FLOAT],
            parallel=True,       
            batch=False,         
            prefetch_queue_depth=8 
        )
        
        # Move to GPU
        images = self.images.gpu()
        masks = self.masks.gpu()
        matrices = self.matrices 
        
        # 1. Geometric Augmentation (Warp Affine)
        # Matrix comes from CPU (controlled by parsed params)
        images = fn.warp_affine(images, matrix=matrices, interp_type=types.INTERP_LINEAR, fill_value=0)
        masks = fn.warp_affine(masks, matrix=matrices, interp_type=types.INTERP_NN, fill_value=0)
        
        # 3. Horizontal Flip
        mirror = None
        if self.parser.use_hflip:
            mirror = fn.random.coin_flip(probability=self.parser.hflip_prob)
        
        # 4. Normalize (and Flip)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mirror=mirror, 
            mean=self.parser.mean, # Parsed mean
            std=self.parser.std    # Parsed std
        )
        
        # Masks: (H, W, C) -> (C, H, W)
        if mirror is not None:
            masks = fn.flip(masks, horizontal=mirror)
            
        masks = fn.transpose(masks, perm=[2, 0, 1])
        masks = fn.cast(masks, dtype=types.FLOAT)
        
        return images, masks

class DaliDataLoaderWrapper:
    def __init__(self, pipeline, size, batch_size):
        self.pipeline = pipeline
        self.size = size
        self.batch_size = batch_size
        self.iterator = DALIGenericIterator(
            pipelines=[pipeline], 
            output_map=["image", "mask"], 
            size=size,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL
        )
        
    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = next(self.iterator)
            batch = data[0]
            return batch["image"], batch["mask"]
        except StopIteration:
            self.iterator.reset()
            raise StopIteration

def get_dali_loader(is_train=True, batch_size=None):
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
        
    e_source = XRayExternalSource(is_train=is_train)
    num_workers = getattr(Config, 'NUM_WORKERS', 8)
    
    pipe = XRayDaliPipeline(
        batch_size=batch_size, 
        num_threads=4, 
        device_id=0, 
        external_source=e_source,
        py_num_workers=num_workers
    )
    pipe.build()
    
    return DaliDataLoaderWrapper(pipe, size=len(e_source), batch_size=batch_size)

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
