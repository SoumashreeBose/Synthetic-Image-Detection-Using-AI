"""
dataset.py — Dataset loading, splitting, and Data Loader creation
"""

import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import CFG

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────
# PATH COLLECTION
# ─────────────────────────────────────────────────────────────
def detect_dataset_structure(dataset_root_path):
    """
    Analyzes the dataset root path to determine its folder structure for splits.

    Args:
        dataset_root_path (str): The path to the root directory of the dataset.

    Returns:
        str: 'full_split' if 'train', 'val', 'test' folders exist.
             'train_test_only' if 'train' and 'test' folders exist but 'val' doesn't.
             'single_folder' if none of the above (all data in root or other subfolders).
             Returns None if the root path does not exist.
    """
    if not os.path.isdir(dataset_root_path):
        # Allow passing paths that don't exist yet, handle gracefully later
        # For this function, if it doesn't exist, we can't detect structure.
        return None

    # Check for presence of common split folders (case-insensitive)
    subdirs = [d.lower() for d in os.listdir(dataset_root_path)
               if os.path.isdir(os.path.join(dataset_root_path, d))]
    print(f"Detected subdirectories: {subdirs}")

    has_train = "train" in subdirs
    has_val   = "valid" in subdirs or "val" in subdirs or "validation" in subdirs # Common variations
    has_test  = "test" in subdirs

    if has_train and has_test and has_val :
        return 'full_split'
    elif has_train and has_test:
        return 'train_test_only'
    elif has_test and has_val:
        return 'val_test_only'
    elif has_train and has_val:
        return 'train_val_only'
    else:
        return 'single_folder' # Assume all data is in the root or will be split from a single pool

print(f"Dataset structure detector defined. Example CFG.DATA_ROOT: {CFG.DATA_ROOT}")


def collect_paths(root, sub_folder_name):
    samples = []
    #lowered = [folder.lower() for folder in os.listdir(root)]

    #print(f"{lowered}")
    # Construct the base path based on sub_folder_name
    if sub_folder_name in os.listdir(root):
        base_path = os.path.join(root, sub_folder_name)
    elif sub_folder_name == None:
        base_path = root

    #if not os.path.isdir(base_path):
    #    raise FileNotFoundError(f"Base path not found: {base_path}")
    print(f"Base path: {base_path}")

    found_real_folder = None
    found_fake_folder = None

    # Check immediate subdirectories for 'real' and 'fake' (case-insensitive)
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for subdir in subdirs:
        lower_subdir = subdir.lower()
        if lower_subdir == "real" and found_real_folder is None:
            found_real_folder = os.path.join(base_path, subdir)
        elif lower_subdir == "fake" and found_fake_folder is None:
            found_fake_folder = os.path.join(base_path, subdir)

    if found_real_folder is None:
        raise FileNotFoundError(f"Missing 'real' folder (case-insensitive) in {base_path}")
    if found_fake_folder is None:
        raise FileNotFoundError(f"Missing 'fake' folder (case-insensitive) in {base_path}")

    # Now collect samples using the found folders
    for folder, label_idx in [(found_real_folder, 0), (found_fake_folder, 1)]:
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in EXTS:
                samples.append((os.path.join(folder, f), label_idx))
    random.shuffle(samples)
    return samples


'''def collect_paths(root):
    """
    Walks root/real/ and root/fake/ and returns list of
    (image_path, label) tuples.  real=0, fake=1.
    """
    samples = []
    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        folder = os.path.join(root, label_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                f"Make sure your dataset has real/ and fake/ subfolders."
            )
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in EXTS:
                samples.append((os.path.join(folder, f), label_idx))
    random.shuffle(samples)
    return samples
'''


def split_dataset(samples):
    """Splits into train / val / test."""
    n      = len(samples)
    n_test = int(n * CFG.TEST_SPLIT)
    n_val  = int(n * CFG.VAL_SPLIT)
    train  = samples[n_test + n_val:]
    val    = samples[n_test:n_test + n_val]
    test   = samples[:n_test]
    return train, val, test


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────

def get_spatial_transform(split="train"):
    """
    For EfficientNet-B4 and Swin-Tiny.
    Uses ImageNet normalization.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),
            transforms.RandomCrop(CFG.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_freq_transform(split="train"):
    """
    For FreqCNN — NO ImageNet normalization.
    SRM filters need raw pixel values in [0, 1].
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE + 16, CFG.IMG_SIZE + 16)),
            transforms.RandomCrop(CFG.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
    ])


# ─────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────
# LOADER FACTORY
# ─────────────────────────────────────────────────────────────


def make_loaders(dataset_root_path=CFG.DATA_ROOT):
     # Determine dataset structure
    structure = detect_dataset_structure(dataset_root_path)
    print(f"Detected dataset structure for '{dataset_root_path}': {structure}")

    train_samples, val_samples, test_samples = [], [], []

    def find_val_dir(root):
        for name in ["val", "valid", "validation"]:
            if os.path.isdir(os.path.join(root, name)):
                return name
        return None    

    if structure == 'full_split':
        print("Loading data from pre-defined train, val, test folders.")
        train_samples = collect_paths(dataset_root_path, "train")
        val_dir = find_val_dir(dataset_root_path)
        val_samples = collect_paths(dataset_root_path, val_dir)
        test_samples  = collect_paths(dataset_root_path, "test")
    elif structure == 'train_test_only':
        print("Loading data from pre-defined train and test folders, splitting train for validation.")
        all_train_samples = collect_paths(dataset_root_path, "train")
        train_samples, val_samples, _ = split_dataset(all_train_samples, test_split=CFG.VAL_SPLIT, val_split=0) # Split train into train/val, no new test
        test_samples      = collect_paths(dataset_root_path, "test")
    elif structure == 'val_test_only':
        print("Loading data from pre-defined validation and test folders, splitting validation for train.")
        val_dir = find_val_dir(dataset_root_path)
        all_train_samples = collect_paths(dataset_root_path, val_dir)
        train_samples, val_samples, _ = split_dataset(all_train_samples, test_split=CFG.VAL_SPLIT, val_split=0) # Split train into train/val, no new test
        test_samples      = collect_paths(dataset_root_path, "test")
    elif structure == 'train_val_only':
        print("Loading data from pre-defined train and validation folders, splitting train for test.")
        all_train_samples = collect_paths(dataset_root_path, "train")
        train_samples, test_samples, _ = split_dataset(all_train_samples, test_split=CFG.TEST_SPLIT, val_split=0) # Split train into train/val, no new test
        val_dir = find_val_dir(dataset_root_path)
        val_samples = collect_paths(dataset_root_path, val_dir)
    elif structure == 'single_folder':
        print("Loading all data from a single folder, then splitting into train, val, test.")
        all_samples = collect_paths(dataset_root_path, None)
        train_samples, val_samples, test_samples = split_dataset(all_samples)
    else: # Path not found or unknown structure, default to single folder behavior with empty splits
        print(f"Warning: Dataset path '{dataset_root_path}' not found or structure unknown. Initializing empty loaders.")
        return None, None, None, None, None, None, [] # Return empty if path does not exist


    labels  = [s[1] for s in train_samples]
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler_train = WeightedRandomSampler(weights, len(weights))
    kw = dict(num_workers=CFG.NUM_WORKERS, pin_memory=True)

    train_loader = DataLoader(
        DeepfakeDataset(train_samples, get_spatial_transform("train")),
        batch_size=CFG.BATCH_SIZE, sampler=sampler_train, **kw)
    val_loader = DataLoader(
        DeepfakeDataset(val_samples, get_spatial_transform("val")),
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)
    test_loader = DataLoader(
        DeepfakeDataset(test_samples, get_spatial_transform("test")),
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)

    # Frequency loaders (no normalization)
    train_freq_loader = DataLoader(
        DeepfakeDataset(train_samples, get_freq_transform("train")),
        batch_size=CFG.BATCH_SIZE, sampler=sampler_train, **kw)
    val_freq_loader = DataLoader(
        DeepfakeDataset(val_samples, get_freq_transform("val")),
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)
    test_freq_loader = DataLoader(
        DeepfakeDataset(test_samples, get_freq_transform("test")),
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)

    print(f"✓ Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")
    print(f"✓ real={counts[0]} | fake={counts[1]}")
    return (train_loader, val_loader, test_loader,
            train_freq_loader, val_freq_loader, test_freq_loader,
            test_samples)

(train_loader, val_loader, test_loader,
 train_freq_loader, val_freq_loader, test_freq_loader,
 test_samples) = make_loaders()


"""
def make_loaders(data_root=None):
    """
   # Creates all six DataLoaders needed for training:
   # spatial (train/val/test) and frequency (train/val/test).
"""
    if data_root is None:
        data_root = CFG.DATA_ROOT

    samples                   = collect_paths(data_root)
    train_s, val_s, test_s   = split_dataset(samples)

    labels  = [s[1] for s in train_s]
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    kw_shuffle = dict(
        num_workers=CFG.NUM_WORKERS, pin_memory=(CFG.DEVICE == "cuda"))
    kw_fixed   = dict(
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw_shuffle)

    train_loader = DataLoader(
        DeepfakeDataset(train_s, get_spatial_transform("train")),
        batch_size=CFG.BATCH_SIZE, sampler=sampler, **kw_shuffle)
    val_loader = DataLoader(
        DeepfakeDataset(val_s, get_spatial_transform("val")), **kw_fixed)
    test_loader = DataLoader(
        DeepfakeDataset(test_s, get_spatial_transform("test")), **kw_fixed)

    train_freq_loader = DataLoader(
        DeepfakeDataset(train_s, get_freq_transform("train")),
        batch_size=CFG.BATCH_SIZE, sampler=sampler, **kw_shuffle)
    val_freq_loader = DataLoader(
        DeepfakeDataset(val_s, get_freq_transform("val")), **kw_fixed)
    test_freq_loader = DataLoader(
        DeepfakeDataset(test_s, get_freq_transform("test")), **kw_fixed)

    print(f"[Data] Train={len(train_s)} | Val={len(val_s)} | Test={len(test_s)}")
    print(f"[Data] real={counts[0]} | fake={counts[1]}")

    return (train_loader, val_loader, test_loader,
            train_freq_loader, val_freq_loader, test_freq_loader,
            test_s)

"""
def make_cross_loader(data_root):
    """
   # Builds spatial + frequency loaders for a cross-dataset path.
   # No train/val split — uses all images for evaluation.
    """
    samples = collect_paths(data_root,"test")
    counts  = np.bincount([s[1] for s in samples])

    kw = dict(batch_size=CFG.BATCH_SIZE, shuffle=False,
              num_workers=CFG.NUM_WORKERS,
              pin_memory=(CFG.DEVICE == "cuda"))

    sp_loader = DataLoader(
        DeepfakeDataset(samples, get_spatial_transform("test")), **kw)
    fr_loader = DataLoader(
        DeepfakeDataset(samples, get_freq_transform("test")), **kw)

    print(f"  Size={len(samples)} | real={counts[0]} | fake={counts[1]}")
    return sp_loader, fr_loader

def detect_class_folders(search_root):
    """
    STEP 2: Given a folder, finds real/ and fake/ subfolders
    regardless of what they are named.

    Returns:
        (real_path, fake_path) or (None, None) if not found.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    REAL_NAMES = {
        "real", "real_images", "reals", "authentic",
        "original", "originals", "genuine", "real_faces",
        "0", "class0", "class_0", "neg", "negative",
        "true", "natural", "real_vs_fake/real"
    }
    FAKE_NAMES = {
        "fake", "fake_images", "fakes", "synthetic",
        "manipulated", "generated", "forged", "deepfake",
        "fake_faces", "ai_generated", "tampered", "altered",
        "1", "class1", "class_1", "pos", "positive",
        "false", "artificial"
    }

    def count_images(folder):
        try:
            return sum(
                1 for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in image_exts)
        except Exception:
            return 0

    if not os.path.isdir(search_root):
        return None, None

    subdirs = {
        d.lower(): os.path.join(search_root, d)
        for d in os.listdir(search_root)
        if os.path.isdir(os.path.join(search_root, d))
    }

    real_path = next(
        (path for name, path in subdirs.items()
         if name in REAL_NAMES), None)
    fake_path = next(
        (path for name, path in subdirs.items()
         if name in FAKE_NAMES), None)

    # Validate they actually contain images
    if real_path and fake_path:
        if count_images(real_path) > 0 and count_images(fake_path) > 0:
            return real_path, fake_path

    return None, None

def collect_paths_auto(root):
    """
    Combines both detectors:
    1. detect_dataset_structure → finds which split subfolder to use
    2. detect_class_folders     → finds real/ and fake/ inside it

    Returns list of (image_path, label) tuples.
    real=0, fake=1
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"\n  Scanning : {root}")

    # ── STEP 1: Detect split structure ───────────────────────
    split_structure = detect_dataset_structure(root)
    print(f"  Split structure : {split_structure}")

    # Build list of candidate folders to search for real/fake
    search_candidates = []

    if split_structure == "full_split":
        # Has train/ val/ test/ — use train/ for training data
        # We collect from all three and let make_loaders split
        # Or use train/ only for multi-dataset training
        for split_name in ["train", "val", "test", "valid", "validation"]:
            for d in os.listdir(root):
                if d.lower() == split_name:
                    search_candidates.append(
                        os.path.join(root, d))
                    break
        # Fallback: also try root itself
        search_candidates.append(root)

    elif split_structure == "train_test_only":
        for split_name in ["train", "test"]:
            for d in os.listdir(root):
                if d.lower() == split_name:
                    search_candidates.append(
                        os.path.join(root, d))
                    break
        search_candidates.append(root)

    elif split_structure in ("val_test_only", "train_val_only"):
        # Use whatever split folders exist
        for d in os.listdir(root):
            search_candidates.append(os.path.join(root, d))
        search_candidates.append(root)

    else:
        # single_folder: real/fake are directly inside root
        search_candidates = [root]

    # ── STEP 2: Find real/fake inside candidates ──────────────
    all_samples = []
    found_any   = False

    for candidate in search_candidates:
        if not os.path.isdir(candidate):
            continue

        real_path, fake_path = detect_class_folders(candidate)

        if real_path and fake_path:
            print(f"  Found classes in : {candidate}")
            print(f"    real → {os.path.basename(real_path)}/")
            print(f"    fake → {os.path.basename(fake_path)}/")
            found_any = True

            for f in os.listdir(real_path):
                if os.path.splitext(f)[1].lower() in image_exts:
                    all_samples.append(
                        (os.path.join(real_path, f), 0))

            for f in os.listdir(fake_path):
                if os.path.splitext(f)[1].lower() in image_exts:
                    all_samples.append(
                        (os.path.join(fake_path, f), 1))

    # ── STEP 3: Handle flat folder with filename prefixes ─────
    if not found_any:
        real_kws = ["real", "authentic", "original"]
        fake_kws = ["fake", "synthetic", "generated",
                    "manipulated"]
        flat_samples = []
        for f in os.listdir(root):
            if os.path.splitext(f)[1].lower() not in image_exts:
                continue
            fl = f.lower()
            if any(k in fl for k in real_kws):
                flat_samples.append(
                    (os.path.join(root, f), 0))
            elif any(k in fl for k in fake_kws):
                flat_samples.append(
                    (os.path.join(root, f), 1))

        if flat_samples:
            print(f"  Found flat folder with filename prefixes")
            all_samples.extend(flat_samples)
            found_any = True

    # ── STEP 4: Nothing found — print tree and raise ──────────
    if not all_samples:
        print(f"\n  [Structure scan of {root}]")
        for item in sorted(os.listdir(root))[:20]:
            full = os.path.join(root, item)
            if os.path.isdir(full):
                n = len(os.listdir(full))
                print(f"    DIR  {item}/  ({n} items)")
            else:
                print(f"    FILE {item}")
        raise ValueError(
            f"\nCould not detect real/fake structure in: {root}\n"
            f"Split structure detected: {split_structure}\n"
            f"Expected real/ and fake/ folders (or known equivalents).\n"
            f"Add your folder names to REAL_NAMES or FAKE_NAMES "
            f"inside detect_class_folders()."
        )

    # Deduplicate in case same images were found in multiple splits
    all_samples = list(dict.fromkeys(all_samples))
    random.shuffle(all_samples)

    counts = np.bincount([s[1] for s in all_samples])
    print(f"  ✓ real={counts[0]} | fake={counts[1]} | "
          f"total={len(all_samples)}")
    return all_samples


def make_combined_loader(dataset_paths, split="train"):
    """
    Merges multiple datasets into a single loader.
    Auto-detects both split structure and class folder names.
    """
    all_samples = []

    print(f"\n{'='*55}")
    print(f"  Building combined loader  ({split})")
    print(f"{'='*55}")

    for path in dataset_paths:
        if not os.path.isdir(path):
            print(f"\n  [SKIP] Not found: {path}")
            continue
        try:
            samples = collect_paths_auto(path)
            all_samples.extend(samples)
        except ValueError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [SKIP] Unexpected error in {path}: {e}")

    if not all_samples:
        raise ValueError(
            "No valid datasets found. "
            "Check paths in CFG.MULTI_TRAIN_DATASETS."
        )

    # Deduplicate across datasets
    all_samples = list(dict.fromkeys(all_samples))
    random.shuffle(all_samples)

    labels  = [s[1] for s in all_samples]
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    print(f"\n  ✓ Combined total : {len(all_samples)} images")
    print(f"    real={counts[0]} | fake={counts[1]}")

    kw = dict(
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(CFG.DEVICE == "cuda")
    )
    transform = (get_spatial_transform("train")
                 if split == "train"
                 else get_spatial_transform("val"))

    if split == "train":
        return DataLoader(
            DeepfakeDataset(all_samples, transform),
            batch_size=CFG.BATCH_SIZE,
            sampler=sampler, **kw
        )
    return DataLoader(
        DeepfakeDataset(all_samples, transform),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False, **kw
    )
"""
def make_combined_loader(dataset_paths, split="train"):
    """
    #Merges multiple datasets into a single loader for
    #multi-dataset training experiment.
"""
    all_samples = []
    for path in dataset_paths:
        if not os.path.isdir(path):
            print(f"  [SKIP] {path} not found")
            continue
        try:
            samples = collect_paths(path)
            all_samples.extend(samples)
            counts = np.bincount([s[1] for s in samples])
            print(f"  Added {os.path.basename(path)}: "
                  f"real={counts[0]} fake={counts[1]}")
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")

    if not all_samples:
        raise ValueError("No valid datasets found in dataset_paths.")

    random.shuffle(all_samples)
    labels  = [s[1] for s in all_samples]
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    print(f"\n  Combined: {len(all_samples)} images | "
          f"real={counts[0]} fake={counts[1]}")

    kw = dict(num_workers=CFG.NUM_WORKERS,
              pin_memory=(CFG.DEVICE == "cuda"))
    transform = (get_spatial_transform("train") if split == "train"
                 else get_spatial_transform("val"))

    if split == "train":
        return DataLoader(
            DeepfakeDataset(all_samples, transform),
            batch_size=CFG.BATCH_SIZE, sampler=sampler, **kw)
    return DataLoader(
        DeepfakeDataset(all_samples, transform),
        batch_size=CFG.BATCH_SIZE, shuffle=False, **kw)

        """