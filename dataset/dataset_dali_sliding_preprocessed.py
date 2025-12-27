
import os
import cv2
import numpy as np
import torch
import random
import albumentations as A
from sklearn.model_selection import GroupKFold
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from config import Config

# ============================================================
# [SLIDING WINDOW VERSION V3]
# Offline Preprocessed Data Loader
# No CPU CLAHE, No JSON Parsing at Runtime
# ============================================================

def get_transforms(is_train=True):
    # Only geometric augmentations are needed now
    if is_train:
        return A.Compose([
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
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

class DaliTransformParser:
    def __init__(self, is_train=True):
        self.is_train = is_train
        transforms = get_transforms(is_train)
        
        self.window_size = getattr(Config, 'WINDOW_SIZE', 1024)
        self.stride = getattr(Config, 'STRIDE', 1024)
        
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        
        self.use_geometric = False
        self.geo_prob = 0.0
        self.rotate_limit = 0; self.scale_limit = 0; self.shift_limit = 0
        self.use_hflip = False; self.hflip_prob = 0.0

        if transforms:
            t_list = transforms.transforms if isinstance(transforms, A.Compose) else transforms
            for t in t_list:
                if isinstance(t, A.Normalize):
                    self.mean = [m * 255 for m in t.mean]
                    self.std = [s * 255 for s in t.std]
                elif isinstance(t, A.ShiftScaleRotate):
                    self.use_geometric = True
                    self.geo_prob = t.p
                    r_lim = max(abs(t.rotate_limit[0]), abs(t.rotate_limit[1]))
                    s_lim = max(abs(t.scale_limit[0]), abs(t.scale_limit[1]))
                    sx = max(abs(t.shift_limit_x[0]), abs(t.shift_limit_x[1]))
                    sy = max(abs(t.shift_limit_y[0]), abs(t.shift_limit_y[1]))
                    self.rotate_limit = r_lim
                    self.scale_limit = s_lim
                    self.shift_limit = max(sx, sy)
                elif isinstance(t, A.HorizontalFlip):
                    self.use_hflip = True
                    self.hflip_prob = t.p

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
        self.parser = DaliTransformParser(is_train)
        
        # New Offline Data Paths
        self.image_root = "../data/train/DCM_CLAHE"
        self.mask_root = "../data/train/masks_png"
        
        if not os.path.exists(self.image_root) or not os.path.exists(self.mask_root):
            raise FileNotFoundError("Offline data not found. Run tools/prepare_sliding_dataset.py first.")
            
        # Scan files
        imgs = sorted(os.listdir(self.image_root))
        
        _filenames = np.array(imgs)
        # Assuming label filenames match image filenames (except extension)
        # Images are .jpg, Masks are .png
        
        groups = [os.path.dirname(fname) for fname in _filenames] # Actually flat directory
        # Just use filename as group since we don't have subdirs anymore
        groups = _filenames 
        ys = [0 for _ in _filenames]
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        val_fold = getattr(Config, 'VAL_FOLD', 4)

        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if (i == val_fold) ^ is_train:
                filenames += list(_filenames[y])
        
        self.filenames = list(filenames)
        
        # Sliding Window
        self.window_size = self.parser.window_size
        self.stride = self.parser.stride
        self.patches_per_image = ((2048 - self.window_size) // self.stride + 1) ** 2
        
        self.n = len(self.filenames) * self.patches_per_image
        print(f"[Sliding Window V3] Images: {len(self.filenames)}, Patches per image: {self.patches_per_image}, Total: {self.n}")
        
        self.indices = list(range(len(self.filenames)))
        if is_train: random.shuffle(self.indices)
            
        # Cache
        self.last_image_idx = -1
        self.cached_image = None
        self.cached_mask = None # cached mask (index map H,W)

    def __len__(self):
        return self.n

    def __call__(self, sample_info):
        idx_in_epoch = sample_info.idx_in_epoch
        if idx_in_epoch >= self.n: idx_in_epoch %= self.n
        
        image_idx = idx_in_epoch // self.patches_per_image
        patch_idx = idx_in_epoch % self.patches_per_image
        
        real_idx = self.indices[image_idx]
        
        if self.last_image_idx == real_idx and self.cached_image is not None:
            image = self.cached_image
            mask = self.cached_mask
        else:
            image_name = self.filenames[real_idx]
            mask_name = os.path.splitext(image_name)[0] + ".png"
            
            # Read Image
            image_path = os.path.join(self.image_root, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read Mask (Index PNG, Loading as Grayscale)
            mask_path = os.path.join(self.mask_root, mask_name)
            mask_indices = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # (2048, 2048) values 0..29
            
            # Convert Index Mask to Multi-channel Mask (One-Hot-like)
            # This is slow in Python? 
            # DALI pipeline expects (H, W, C) mask? 
            # Wait, our previous code generated (2048, 2048, 29).
            # We should probably do this conversion here OR optimize DALI to take index map.
            # For back-compatibility with previous pipeline, let's generate (2048, 2048, 29).
            # Optimization: slice it first, then expand? No, strict patch logic.
            
            # Let's verify how previous code worked.
            # It created `label = np.zeros(label_shape, dtype=np.uint8)` and fillPoly.
            
            # Here:
            # mask_indices: (2048, 2048)
            # We need patch (1024, 1024, 29).
            
            # Let's CACHE the Index Mask (much smaller), and expand ONLY the patch.
            self.last_image_idx = real_idx
            self.cached_image = image
            self.cached_mask = mask_indices
            mask = mask_indices

        # Crop Patch
        patches_per_row = (2048 - self.window_size) // self.stride + 1
        row = patch_idx // patches_per_row
        col = patch_idx % patches_per_row
        crop_y = row * self.stride
        crop_x = col * self.stride
        
        image_patch = image[crop_y:crop_y+self.window_size, crop_x:crop_x+self.window_size]
        mask_patch_indices = mask[crop_y:crop_y+self.window_size, crop_x:crop_x+self.window_size]
        
        # Expand Mask Patch to Channels
        # Use simple numpy broadcasting or indexing
        # mask_patch_indices has values 0..29. 0 is BG. 1..29 are classes.
        # We need output (1024, 1024, 29). Channel 0 corresponds to class index 1 ('finger-1').
        
        # Fast one-hot encoding
        # This can be slow if not careful.
        # classes = 29.
        
        label_patch = np.zeros((self.window_size, self.window_size, len(Config.CLASSES)), dtype=np.uint8)
        
        # Only relevant indices
        unique_indices = np.unique(mask_patch_indices)
        for idx in unique_indices:
            if idx > 0: # 0 is background
                label_patch[..., idx-1] = (mask_patch_indices == idx).astype(np.uint8)
                
        matrix = self.parser.get_affine_matrix(self.window_size, self.window_size)
        return image_patch, label_patch, matrix

    def shuffle(self):
        if self.is_train: random.shuffle(self.indices)

class XRayDaliPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_source, py_num_workers=1):
        super(XRayDaliPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12, py_start_method="spawn", py_num_workers=py_num_workers
        )
        self.source = external_source
        self.parser = external_source.parser 
        
    def define_graph(self):
        self.images, self.masks, self.matrices = fn.external_source(
            source=self.source, num_outputs=3, dtype=[types.UINT8, types.UINT8, types.FLOAT],
            parallel=True, batch=False, prefetch_queue_depth=8 
        )
        
        images = self.images.gpu() 
        masks = self.masks.gpu() 
        matrices = self.matrices 
        
        # No CLAHE here (already done offline)
        
        if self.parser.use_geometric:
             images = fn.warp_affine(images, matrix=matrices, interp_type=types.INTERP_LINEAR, fill_value=0)
             masks = fn.warp_affine(masks, matrix=matrices, interp_type=types.INTERP_NN, fill_value=0)
        
        mirror = None
        if self.parser.use_hflip:
             mirror = fn.random.coin_flip(probability=self.parser.hflip_prob)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, output_layout="CHW", mirror=mirror,
            mean=self.parser.mean, std=self.parser.std    
        )
        
        if mirror is not None:
             masks = fn.flip(masks, horizontal=mirror)
             
        masks = fn.transpose(masks, perm=[2, 0, 1])
        masks = fn.cast(masks, dtype=types.FLOAT)
        
        return images, masks

class DaliDataLoaderWrapper:
    def __init__(self, pipeline, source, batch_size):
        self.pipeline = pipeline
        self.source = source
        self.batch_size = batch_size
        self.iterator = DALIGenericIterator(
            pipelines=[pipeline], output_map=["image", "mask"], 
            size=len(source), auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL
        )
    def __len__(self): return (len(self.source) + self.batch_size - 1) // self.batch_size
    def __iter__(self): return self
    def __next__(self):
        try:
            data = next(self.iterator)
            batch = data[0]
            return batch["image"], batch["mask"]
        except StopIteration:
            self.source.shuffle()
            raise StopIteration

def get_dali_loader(is_train=True, batch_size=None):
    if batch_size is None: batch_size = Config.BATCH_SIZE
    e_source = XRayExternalSource(is_train=is_train)
    num_workers = getattr(Config, 'NUM_WORKERS', 8)
    dali_threads = getattr(Config, 'DALI_NUM_THREADS', 4) 
    device_id = getattr(Config, 'DEVICE_ID', 0)
    
    pipe = XRayDaliPipeline(
        batch_size=batch_size, num_threads=dali_threads, device_id=device_id, 
        external_source=e_source, py_num_workers=num_workers
    )
    pipe.build()
    return DaliDataLoaderWrapper(pipe, source=e_source, batch_size=batch_size)

# ============================================================
# Inference Dataset (Sliding Window) - V3 (Also reads offline if avail? No, keeping orig for simplicity)
# Or should we update? Inference data is test set, offline proc was only for Train.
# Let's keep Test set usage as V2/V1 logic (load PNG, do CLAHE). 
# But wait, logic above points to ../data/train.
# So InferenceDataset below is for TEST set and should utilize same logic as V2 (Runtime CLAHE).
# ============================================================
class XRayInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.image_root = Config.TEST_IMAGE_ROOT
        self.filenames = np.array(sorted([
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ]))
        self.transforms = transforms
    def __len__(self): return len(self.filenames)
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).float(), image_name
