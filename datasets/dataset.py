import os
import cv2
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir, annotations_file, augmentations=None, target_size=(800, 800)):
        self.images_dir = root_dir
        self.augmentations = augmentations
        self.target_size = target_size
        # self.augmentations = augmentations

        # Read the annotations file
        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)
        
        # Extract the images, annotations, and categories
        self.images = data.get('images', [])
        self.annotations = data.get('annotations', [])
        self.categories = data.get('categories', [])
        
    def __len__(self):
        return len(self.images)
    
    def get_categories(self):
        return {idx: label['name'] for idx, label in enumerate(self.categories)}
    
    def get_category_number(self):
        return len(self.categories)

    def load_image(self, idx: int):
        filename = self.images[idx]['file_name']
        path = os.path.join(self.images_dir, filename)
        image = cv2.imread(path)
        return image, path
    

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes = self.load_target(idx)
        
        h, w, _ = image.shape
        #_ , w, h = image.shape
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }

        # Apply augmentations        
        if self.augmentations:
            transformed = self.augmentations(
                image=image, 
                bboxes=boxes, 
                category_ids=labels
            )
            image = torch.as_tensor(transformed['image'].astype("float32").transpose(2, 0, 1))
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
        else:
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # Convert lists to numpy arrays before creating tensors
        boxes = np.array(boxes)
        labels = np.array(labels)

        # Pad bounding boxes and labels to a fixed size based on the number of annotations
        num_boxes = len(boxes)
        padded_boxes = torch.tensor(boxes).float()
        padded_labels = torch.tensor(labels).float()

        # Pad to the maximum number of boxes
        if num_boxes < self.max_boxes:
            pad_boxes = torch.zeros((self.max_boxes - num_boxes, 4))
            pad_labels = torch.full((self.max_boxes - num_boxes,), -1)
            padded_boxes = torch.cat([padded_boxes, pad_boxes], dim=0)
            padded_labels = torch.cat([padded_labels, pad_labels], dim=0)
        
        return image, padded_boxes, padded_labels, metadata

  
class BoxDataset(BaseDataset):

    def __init__(self, root_dir, annotations_file, augmentations=None, target_size=(800, 800)):
        super().__init__(root_dir, annotations_file, augmentations, target_size)
        self.max_boxes = 500
    
    def load_target(self, idx: int):
        image_id = self.images[idx]['id']
        annotations = [ann for ann in self.annotations if ann['image_id'] == image_id]

        labels = []
        boxes = []
        for annotation in annotations:
            if annotation["bbox"][-1] < 1 or annotation["bbox"][-2] < 1:
                continue
            
            labels.append(annotation["category_id"])
            boxes.append(annotation["bbox"] if "bbox" in annotation else [])  
        return labels, boxes

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes = self.load_target(idx)
        
        h, w, _ = image.shape
        #_ , w, h = image.shape
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }

        # Apply augmentations        
        if self.augmentations:
            transformed = self.augmentations(
                image=image, 
                bboxes=boxes, 
                category_ids=labels
            )
            image = torch.as_tensor(transformed['image'].astype("float32").transpose(2, 0, 1))
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
        else:
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # Convert lists to numpy arrays before creating tensors
        boxes = np.array(boxes)
        labels = np.array(labels)

        # Pad bounding boxes and labels to a fixed size based on the number of annotations
        num_boxes = len(boxes)
        padded_boxes = torch.tensor(boxes).float()
        padded_labels = torch.tensor(labels).float()

        # Pad to the maximum number of boxes
        if num_boxes < self.max_boxes:
            pad_boxes = torch.zeros((self.max_boxes - num_boxes, 4))
            pad_labels = torch.full((self.max_boxes - num_boxes,), -1)
            padded_boxes = torch.cat([padded_boxes, pad_boxes], dim=0)
            padded_labels = torch.cat([padded_labels, pad_labels], dim=0)
        
        return image, padded_boxes, padded_labels, metadata