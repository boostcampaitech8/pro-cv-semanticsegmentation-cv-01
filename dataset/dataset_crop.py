import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from config import Config

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=Config.RESIZE_SIZE[0], width=Config.RESIZE_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Resize(Config.RESIZE_SIZE[0], Config.RESIZE_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

def get_label_bbox(annotations, padding=50):
    all_points = []
    for ann in annotations:
        points = np.array(ann['points'])
        all_points.append(points)
    
    if not all_points:
        return None
        
    all_points = np.vstack(all_points)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h) + padding * 2
    
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    
    return int(cx - side/2), int(cy - side/2), int(cx + side/2), int(cy + side/2)

def get_heuristic_bbox(image, threshold=20, padding=50):
    # thresholding
    binary = (image > threshold).astype(np.uint8)
    
    # remove noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]
    
    # get bounding box covering all contours
    all_cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_cnt)
    
    side = max(w, h) + padding * 2
    cx, cy = x + w/2, y + h/2
    
    return int(cx - side/2), int(cy - side/2), int(cx + side/2), int(cy + side/2)

def crop_with_padding(image, x_min, y_min, x_max, y_max):
    H, W = image.shape[:2]
    target_w, target_h = x_max - x_min, y_max - y_min
    
    # Create black canvas
    if len(image.shape) == 3:
        new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else:
        new_image = np.zeros((target_h, target_w), dtype=np.uint8)
        
    # Valid range in original image
    src_x1, src_y1 = max(0, x_min), max(0, y_min)
    src_x2, src_y2 = min(W, x_max), min(H, y_max)
    
    # Range in new canvas
    dst_x1, dst_y1 = max(0, -x_min), max(0, -y_min)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    if src_x1 < src_x2 and src_y1 < src_y2:
        new_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
    return new_image

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        self.is_train = is_train
        self.transforms = transforms
        
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
        
        image = cv2.imread(image_path)

        label_name = self.labelnames[item]
        label_path = os.path.join(Config.LABEL_ROOT, label_name)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        # 1. Label-based Square Crop
        bbox = get_label_bbox(annotations, padding=50)
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            image = crop_with_padding(image, x_min, y_min, x_max, y_max)
            # adjust annotation points based on crop (including padding shift)
            for ann in annotations:
                ann['points'] = [[p[0]-x_min, p[1]-y_min] for p in ann['points']]

        label_shape = tuple(image.shape[:2]) + (len(Config.CLASSES), )
        mask = np.zeros(label_shape, dtype=np.uint8)
        
        for ann in annotations:
            c = ann["label"]
            class_ind = Config.CLASS2IND[c]
            points = np.array(ann["points"], dtype=np.int32)
            
            # Fix OpenCV Mat layout issue with slices
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            mask[..., class_ind] = class_label

        if self.transforms:
            result = self.transforms(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

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
        orig_h, orig_w = image.shape[:2]
        
        # 1. Heuristic-based Crop
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_min, y_min, x_max, y_max = get_heuristic_bbox(gray)
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        if self.transforms is not None:
            result = self.transforms(image=cropped_image)
            image_tensor = result["image"]
        else:
            image_tensor = cropped_image

        image_tensor = image_tensor.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_tensor).float()

        # crop_info is needed to restore mask in original coords
        crop_info = np.array([y_min, y_max, x_min, x_max, orig_h, orig_w], dtype=np.int32)
        
        return image_tensor.float(), image_name, crop_info
