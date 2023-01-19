import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def get_img_lst(root_dir):
    img_lst = []
    class_name_lst = os.listdir(root_dir)
    for class_name in class_name_lst:
        temp_img_lst = os.listdir(os.path.join(root_dir, class_name))
        for img_name in temp_img_lst:
            img_lst.append(os.path.join(root_dir, class_name,img_name))
    return img_lst
    

class Gray2RGBDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform, normal, gray_normal):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_lst = get_img_lst(self.root_dir)
        self.transform = transform
        self.gray_transform = transforms.Grayscale(num_output_channels=1)
        self.normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.gray_normal = transforms.Normalize(mean=[0.299*0.485+0.587*0.456+0.114*0.406], std=[0.299*0.299+0.587*0.224+0.114*0.225])

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        img_path = self.image_lst[idx]
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        if self.transform:
            image = self.transform(image)
        gray_image = self.gray_transform(image)
        image = self.normal(image)
        gray_image = self.gray_normal(gray_image)
        return image, gray_image