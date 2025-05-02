import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

def collate_fn(batch):
    """Custom collate function to filter out None samples"""
    batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class UnlabeledDataset(Dataset):
    """Dataset for unlabeled images used in DINO pretraining."""
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
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            return None, 0 # Important: return None for both image and dummy label

        if self.transform:
            try:
                crops = self.transform(image)
                # Ensure transform returns a list of tensors
                if not isinstance(crops, list) or not all(isinstance(c, torch.Tensor) for c in crops):
                     raise ValueError("Transform did not return a list of tensors")
                return crops, 0 # Return dummy label 0 for unlabeled data
            except Exception as e:
                print(f"Warning: Failed to transform image {img_path}. Error: {e}")
                return None, 0
        else:
             # Handle case where no transform is provided if necessary
             # Maybe convert PIL image to tensor?
             return image, 0 # Return PIL image and dummy label


class Classification_dataset(Dataset):
    """Dataset for labeled classification task."""
    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes if classes else ['COVID', 'Lung_Opacity', 'Viral_Pneumonia', 'Normal']
        self.data = []
        self.class_to_id = {clazz: i for i, clazz in enumerate(self.classes)}

        for clazz in self.classes:
            class_dir = os.path.join(root_dir, clazz)
            img_dir = os.path.join(class_dir, 'images')
            if not os.path.isdir(img_dir):
                img_dir = class_dir 

            if os.path.isdir(img_dir):
                class_id = self.class_to_id[clazz]
                for img_name in os.listdir(img_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(img_dir, img_name)
                        self.data.append((img_path, class_id))
            else:
                print(f"Warning: Directory not found for class {clazz}: {img_dir}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                raise ValueError(f"Failed to read image {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None 

        if self.transform:
            try:
                transformed = self.transform(image=img)
                img = transformed['image']
                if not isinstance(img, torch.Tensor):
                    raise ValueError("Transform did not return a tensor")
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                return None, None

        return img, label