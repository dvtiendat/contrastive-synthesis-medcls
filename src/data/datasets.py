# src/data/datasets.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2 # If using albumentations for classification dataset

class DINOUnlabeledDataset(Dataset):
    """Dataset for unlabeled images used in self-supervised pretraining."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None 
        if self.transform:
            crops = self.transform(image)
            return crops 

        return image, 0

class ClassificationDataset(Dataset):
    """Dataset for labeled image classification (used for fine-tuning)."""
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.data = []
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}

        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            # Assuming images are directly inside class directories
            # Adjust if structure is different (e.g., has 'images' subfolder)
            img_dir = class_dir #os.path.join(class_dir, 'images') # Adjust if needed

            if not os.path.isdir(img_dir):
                 print(f"Warning: Image directory not found: {img_dir}")
                 continue

            class_id = self.class_to_id[clazz]
            for img_file in os.listdir(img_dir):
                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(img_dir, img_file)
                    self.data.append((img_path, class_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            # Use PIL for torchvision transforms, cv2 for albumentations
            if self.transform and 'albumentations' in str(type(self.transform)):
                 img = cv2.imread(img_path)
                 if img is None:
                     raise ValueError(f"cv2 failed to load image: {img_path}")
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                 img = Image.open(img_path).convert('RGB')

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None # Signal error to collate_fn

        if self.transform:
            if 'albumentations' in str(type(self.transform)):
                 transformed = self.transform(image=img)
                 img_tensor = transformed['image']
            else:
                 img_tensor = self.transform(img)

            # Basic check after transform
            if not isinstance(img_tensor, torch.Tensor):
                 print(f"Warning: Transform did not return a tensor for {img_path}")
                 return None, None

            return img_tensor, label

        return img, label

def build_dino_dataset(data_path, global_crops_scale, local_crops_scale,
                       local_crops_number, global_size, local_size):
    """Helper function to build DINO dataset."""
    transform = DataAugmentationDINO(
        global_crops_scale, local_crops_scale, local_crops_number,
        global_size, local_size
    )
    dataset = UnlabeledDataset(data_path, transform=transform)
    print(f"Built DINO dataset with {len(dataset)} images from {data_path}.")
    return dataset

def collate_fn_filter_none(batch):
    """Collate function that filters out None samples."""
    batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn_dino(batch):
    """Collate function for DINO that handles lists of crops and filters None."""
    batch = [sample for sample in batch if sample is not None and sample[0] is not None]
    if len(batch) == 0:
        return None

    num_crops = len(batch[0]) 
    collated_crops = []
    for i in range(num_crops):
        collated_crops.append(torch.stack([sample[i] for sample in batch]))

    return collated_crops 