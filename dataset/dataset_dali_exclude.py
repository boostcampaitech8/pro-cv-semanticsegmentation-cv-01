
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
# [EXCLUSION LIST]
# ============================================================
EXCLUDE_FILENAMES = [
    "ID058/image1661392103627.png", # 왼손 라벨링 반대로
    "ID325/image1664846270124.png", # 오른손 수근골 아래 라인 라벨링 오류
    "ID363/image1664935962797.png", # 오른손 반지
    "ID547/image1667353928376.png"  # 왼손 Ulna 위에 pisiform 라벨링 오류
]

# ============================================================
# [SINGLE SOURCE OF TRUTH]
# ============================================================
def get_transforms(is_train=True):
    # Matches dataset_final.py exactly
    if is_train:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            
            # CLAHE (CPU)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            
            # SSR (Will be Parsed for GPU & Scaled down)
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
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# ============================================================
# [LOGIC] Parse get_transforms using GPU for SSR
# ============================================================
class DaliTransformParser:
    def __init__(self, is_train=True):
        self.is_train = is_train
        transforms = get_transforms(is_train)
        
        # Defaults
        self.resize_shape = Config.RESIZE_SIZE 
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.cpu_transforms = []
        
        # Init Geometry flags
        self.use_geometric = False
        self.geo_prob = 0.0
        self.rotate_limit = 0; self.scale_limit = 0; self.shift_limit = 0
        self.use_hflip = False; self.hflip_prob = 0.0

        # [CORRECTION FACTOR]
        original_dim = 2048.0
        target_dim = float(Config.RESIZE_SIZE[0])
        self.correction_factor = target_dim / original_dim 
        
        if transforms:
            t_list = transforms.transforms if isinstance(transforms, A.Compose) else transforms
            for t in t_list:
                if isinstance(t, A.Resize):
                    self.resize_shape = (t.height, t.width)
                    # Recalculate factor in case resize changed
                    target_dim = float(t.height)
                    self.correction_factor = target_dim / original_dim
                    
                elif isinstance(t, A.Normalize):
                    self.mean = [m * 255 for m in t.mean]
                    self.std = [s * 255 for s in t.std]
                    
                elif isinstance(t, A.ShiftScaleRotate):
                    self.use_geometric = True
                    self.geo_prob = t.p
                    
                    # [APPLY CORRECTION]
                    r_lim = max(abs(t.rotate_limit[0]), abs(t.rotate_limit[1]))
                    s_lim = max(abs(t.scale_limit[0]), abs(t.scale_limit[1]))
                    sx = max(abs(t.shift_limit_x[0]), abs(t.shift_limit_x[1]))
                    sy = max(abs(t.shift_limit_y[0]), abs(t.shift_limit_y[1]))
                    sh_lim = max(sx, sy)
                    
                    self.rotate_limit = r_lim * self.correction_factor
                    self.scale_limit = s_lim * self.correction_factor
                    self.shift_limit = sh_lim * self.correction_factor
                    
                elif isinstance(t, A.HorizontalFlip):
                    self.use_hflip = True
                    self.hflip_prob = t.p
                elif isinstance(t, (A.Compose, A.OneOf)):
                    pass 
                else:
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
# External Source (Reads Pre-processed JPEG)
# ============================================================
class XRayExternalSource:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.parser = DaliTransformParser(is_train)
        
        # Determine JPEG Root
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
        
        self.filenames = []
        self.labelnames = []
        val_fold = getattr(Config, 'VAL_FOLD', 4)

        # [NEW] Exclude Logic
        exclude_basenames = {os.path.basename(ex) for ex in EXCLUDE_FILENAMES}

        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if (i == val_fold) ^ is_train:
                for f, l in zip(_filenames[y], _labelnames[y]):
                    fname_only = os.path.basename(f)
                    
                    # Check against exclusion list (Check basename)
                    if fname_only in exclude_basenames:
                        continue
                        
                    self.filenames.append(f)
                    self.labelnames.append(l)
        
        self.n = len(self.filenames)
        
        self.indices = list(range(self.n))
        if is_train:
            random.shuffle(self.indices)

        print(f"\n>>> [CHECK] Final Dataset Size (Exclude Version): {len(self.filenames)} images")
        if is_train:
            print(f">>> [CHECK] Excluded {len(_filenames[y]) - len(self.filenames)} images (approx) based on fold split")
        

    def __len__(self):
        return self.n

    def __call__(self, sample_info):
        idx_in_epoch = sample_info.idx_in_epoch
        if idx_in_epoch >= self.n: idx_in_epoch %= self.n
        
        real_idx = self.indices[idx_in_epoch]
            
        image_name = self.filenames[real_idx]
        
        # [MODIFIED] Construct Path to JPEG
        # Replace .png extension with .jpg
        jpeg_name = os.path.splitext(image_name)[0] + ".jpg"
        image_path = os.path.join(self.jpeg_root, jpeg_name)
        
        # 1. Start with JPEG Reading (Fast)
        # Note: We MUST decode on CPU to apply CLAHE via Albumentations
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"JPEG not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB for Albumentations
        
        # DALI Logic from here:
        h_orig, w_orig = 2048, 2048 # Standardized
        h_target, w_target = self.parser.resize_shape 
        
        # [CPU RESIZE OPTIMIZATION]
        if (h_orig, w_orig) != (h_target, w_target):
             image = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        
        # [OPTIMIZATION] Apply CLAHE after resizing (16x faster)
        if self.parser.cpu_compose:
            image = self.parser.cpu_compose(image=image)["image"]
        
        # Prepare Mask at Target Size
        label_name = self.labelnames[real_idx]
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
            
        # Calc Matrix
        matrix = self.parser.get_affine_matrix(h_target, w_target)

        return image, label, matrix
    
    def shuffle(self):
        if self.is_train:
            random.shuffle(self.indices)

# ============================================================
# DALI Pipeline (GPU Acceleration via NVJPEG)
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
        
    def define_graph(self):
        # inputs: 0=JPEG Bytes, 1=Mask(Array), 2=Matrix
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
        
        # Resize moved to CPU to reduce PCIE bandwidth (v4 fix)
        # images = fn.resize(images, ...)
        
        # GPU Affine (ShiftScaleRotate via matrix)
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
