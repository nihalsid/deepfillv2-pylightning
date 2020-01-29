from torch.utils.data import Dataset
from torchvision import transforms
import os
from util import constants
from PIL import Image
import torch
import random
from util.transforms import NormalizeRange


class InpaintDataset(Dataset):

    def __init__(self, root_folder, split, image_size, bbox_shape, bbox_randomness, bbox_margin, bbox_max_num, is_overfit):
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.bbox_shape = bbox_shape
        self.bbox_margin = bbox_margin
        self.bbox_randomness = bbox_randomness
        self.bbox_max_num = bbox_max_num
        with open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.SPLITS_FOLDER, split + ".txt"),"r") as fptr:
            self.files = [x.strip() for x in fptr.readlines() if x.strip() != ""]
            if is_overfit and split == 'train':
                self.files = self.files * 10
        if split == 'train':
            self.transform_func = self.transform_train
        else:
            self.transform_func = self.transform_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.IMAGES_FOLDER, file + ".jpg"))
        image_tensor = self.transform_func(image)
        mask = self.generate_mask()
        return {'name': file, 'image': image_tensor, 'mask': mask}

    def generate_mask(self):
        mask = torch.zeros((1, self.image_size, self.image_size)).float()
        num_bbox = random.randint(1, self.bbox_max_num)
        for i in range(num_bbox):
            bbox_height = random.randint(int(self.bbox_shape * (1 - self.bbox_randomness)), int(self.bbox_shape * (1 + self.bbox_randomness)))
            bbox_width = random.randint(int(self.bbox_shape * (1 - self.bbox_randomness)), int(self.bbox_shape * (1 + self.bbox_randomness)))
            y_min = random.randint(self.bbox_margin, self.image_size - bbox_height - self.bbox_margin - 1)
            x_min = random.randint(self.bbox_margin, self.image_size - bbox_width - self.bbox_margin - 1)
            mask[0, y_min:y_min + bbox_height, x_min:x_min + bbox_width] = 1
        return mask

    def transform_train(self, image):
        transform_list = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(self.image_size, scale=(1.0, 1.25), ratio=(1, 1), interpolation=2),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            NormalizeRange(minval=-1, maxval=1)
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        composed_transform = transforms.Compose(transform_list)
        return composed_transform(image)

    def transform_test(self, image):
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            NormalizeRange(minval=-1, maxval=1)
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        composed_transform = transforms.Compose(transform_list)
        return composed_transform(image)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    root_folder = 'matterport'
    split = 'train'
    image_size = 256
    bbox_shape = 48
    bbox_randomness = 0.25
    bbox_margin = 32
    bbox_max_num = 2
    dataset = InpaintDataset(root_folder, split, image_size, bbox_shape, bbox_randomness, bbox_margin, bbox_max_num, False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()[j]
            mask = sample['mask'].squeeze().numpy()[j]
            image_unnormalized = ((np.transpose(image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(mask)
            plt.show(block=True)
