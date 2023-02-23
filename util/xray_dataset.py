import torch
from torch.utils.data import Dataset
from torchvision import transforms
# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd

class ChestXrayDataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.
    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """
    def __init__(self, image_dir, image_info, split, transform=transforms.ToTensor(), color='L'):
        """
        Args:
        -----
        - transform: A torchvision.transforms object - 
                     transformations to apply to each image
                     (Can be "transforms.Compose([transforms])")
        - color: Specifies image-color format to convert to 
                 (default is L: 8-bit pixels, black and white)
        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset on ieng6
        - image_info: A Pandas DataFrame of the dataset metadata
        - image_filenames: An array of indices corresponding to the images
        - labels: An array of labels corresponding to the each sample
        - classes: A dictionary mapping each disease name to an int between [0, 13]
        """
        
        self.transform = transform
        self.color = color
        self.image_dir = image_dir
        self.image_info = pd.read_csv(image_info)
        # self.image_filenames = self.image_info["Image Index"]
        with open(split, 'r') as f:
            self.image_filenames = f.readlines()
        self.image_filenames = [fileName.strip('\n') for fileName in self.image_filenames]
        self.labels = self.image_info["Finding Labels"]
        self.classes = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema", 
                10: "Emphysema", 11: "Fibrosis", 
                12: "Pleural_Thickening", 13: "Hernia"}
        

        
    def __len__(self):
        
        # Return the total number of data samples
        return len(self.image_filenames)


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get
        Returns:
        --------
        - A tuple (image, label)
        """
        
        # Compose the path to the image file from the image_dir + image_name
        image_path = os.path.join(self.image_dir, self.image_filenames[ind])
        
        # Load the image
        image = Image.open(image_path).convert(mode=str(self.color))

        # If a transform is specified, apply it
        if self.transform is not None:
            image = self.transform(image)
            
        # Verify that image is in Tensor format
        if type(image) is not torch.Tensor:
            image = transforms.ToTensor(image)

        # Convert multi-class label into binary encoding 
        new_ind = self.image_info[self.image_info["Image Index"]==self.image_filenames[ind]].index.tolist()[0]
        label = self.convert_label(self.labels[new_ind], self.classes)
        
        # Return the image and its label
        return (image, label)


    def convert_label(self, label, classes):
        """Convert the numerical label to n-hot encoding.
        
        Params:
        -------
        - label: a string of conditions corresponding to an image's class
        Returns:
        --------
        - binary_label: (Tensor) a binary encoding of the multi-class label
        """
        
        binary_label = torch.zeros(len(classes))
        for key, value in classes.items():
            if value in label:
                binary_label[key] = 1.0
        return binary_label


    def getLabel(self, ind):
        """Returns the label at an index.
        Params:
        -------
        - ind: (int) The index of the image to get
        Returns:
        --------
        - label of image at the index
        """
        
        # Convert multi-class label into binary encoding 
        label = self.convert_label(self.labels[ind], self.classes)
        
        # Return the label at an index
        return label
    