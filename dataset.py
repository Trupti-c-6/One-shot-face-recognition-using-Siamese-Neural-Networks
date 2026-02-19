import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, identity_folders, data_root, transform=None):
        self.identity_folders = identity_folders
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.identity_folders)

    def get_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        folder = self.identity_folders[index]
        folder_path = os.path.join(self.data_root, folder)
        img_paths = os.listdir(folder_path)

        # Anchor
        anchor_idx = random.randint(0, len(img_paths) - 1)
        anchor = self.get_image(os.path.join(folder_path, img_paths[anchor_idx]))

        # Positive
        positive_idx = random.randint(0, len(img_paths) - 1)
        while positive_idx == anchor_idx:
            positive_idx = random.randint(0, len(img_paths) - 1)
        positive = self.get_image(os.path.join(folder_path, img_paths[positive_idx]))

        # Negative
        negative_folder = random.choice(self.identity_folders)
        while negative_folder == folder:
            negative_folder = random.choice(self.identity_folders)

        negative_folder_path = os.path.join(self.data_root, negative_folder)
        negative_img_paths = os.listdir(negative_folder_path)

        negative_idx = random.randint(0, len(negative_img_paths) - 1)
        negative = self.get_image(
            os.path.join(negative_folder_path, negative_img_paths[negative_idx])
        )

        return anchor, positive, negative
      Added FaceDataset implementation
