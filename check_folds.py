
import os
import numpy as np
from sklearn.model_selection import GroupKFold
from config import Config

def check_folds():
    # Simulate the dataset splitting logic
    image_root = Config.IMAGE_ROOT
    
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    
    _filenames = np.array(sorted(pngs))
    groups = [os.path.dirname(fname) for fname in _filenames]
    ys = [0 for fname in _filenames]
    
    gkf = GroupKFold(n_splits=5)
    
    # Target Excludes
    EXCLUDE_FILENAMES = [
        "ID058/image1661392103627.png", 
        "ID325/image1664846270124.png", 
        "ID363/image1664935962797.png", 
        "ID547/image1667353928376.png"  
    ]
    exclude_set = set(EXCLUDE_FILENAMES)
    
    print(f">> Checking Folds for {len(exclude_set)} excluded images...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(_filenames, ys, groups)):
        val_files = _filenames[val_idx]
        
        # Check intersections
        hits = [f for f in val_files if f in exclude_set]
        
        if hits:
            print(f"\n[Fold {fold_idx}] contains {len(hits)} excluded images:")
            for h in hits:
                print(f" - {h}")
        else:
            print(f"[Fold {fold_idx}] Clean.")

if __name__ == "__main__":
    check_folds()
