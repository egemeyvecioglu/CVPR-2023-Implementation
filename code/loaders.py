""" Data loaders for different datasets :
Digits:
    - MNIST
    - SVHN
    - USPS
    - MNIST-M
    - SYN

PACS:
    - Art
    - Cartoon
    - Photo
    - Sketch
    """

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, USPS
from PIL import Image
import numpy as np
from ProgRandConv import ProgRandConvBlock

class MNISTLoader:
    def __init__(self, root, batch_size, num_workers, resize_dim=32):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_dim = resize_dim

    def get_loaders(self):
        train_transform = transforms.Compose(
            [
            transforms.Resize((self.resize_dim, self.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization values
            ]
        )
        dataset = MNIST(root=self.root, train=True, download=True, transform=train_transform)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = MNIST(root=self.root, train=False, download=True, transform=train_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
class SVHNLoader:
    def __init__(self, root, batch_size, num_workers, resize_dim=32):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_dim = resize_dim

    def get_loaders(self):
        train_transform = transforms.Compose(
            [
            transforms.Resize((self.resize_dim, self.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = SVHN(root=self.root, split='train', download=True, transform=train_transform)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = SVHN(root=self.root, split='test', download=True, transform=train_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
class USPSLoader:
    def __init__(self, root, batch_size, num_workers, resize_dim=32):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_dim = resize_dim

    def get_loaders(self):
        train_transform = transforms.Compose(
            [
            transforms.Resize((self.resize_dim, self.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization values
            ]
        )

        dataset = USPS(root=self.root, train=True, download=True, transform=train_transform)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = USPS(root=self.root, train=False, download=True, transform=train_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
    
class MNIST_MLoader:
    def __init__(self) -> None:
        raise NotImplementedError("MNIST-M dataset not implemented")

class SYNLoader:
    def __init__(self, root, batch_size, num_workers):
        raise NotImplementedError("SYN dataset not implemented")

class PACSDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None, train=True):
        self.root_dir = root_dir #data/archive/pacs_label
        self.domain = domain

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if transform is None else transform

        source_file = domain + "_train_kfold.txt" if train else domain + "_test_kfold.txt"
        source_path = os.path.join(self.root_dir, "pacs_label" ,source_file)
        print("Source path:", source_path)
        #source file has the structure for each line: image_path label
        with open(source_path, "r") as f:
            lines = f.readlines()
            self.img_paths = [os.path.join(self.root_dir, "pacs_data/pacs_data/", line.split()[0]) for line in lines]
            self.labels = [int(line.split()[1]) for line in lines]


    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
if __name__ == "__main__":
    # load some images from train and test set of pacs dataset for checking and visualize them
    pacs_train = PACSDataset(root_dir="data/archive", domain="photo", train=True)
    train_loader = DataLoader(pacs_train, batch_size=1, shuffle=False)

    import matplotlib.pyplot as plt

    pacs_test = PACSDataset(root_dir="data/archive", domain="photo", train=False)
    test_loader = DataLoader(pacs_test, batch_size=4, shuffle=False)


    for i, (x, labels) in enumerate(train_loader):
        # show the images

        x, labels = x.to("cpu"), labels.to("cpu")

        model = ProgRandConvBlock(in_channels=x.shape[1], out_channels=3, kernel_size=3, l_max=10, device="cpu", batch_size=x.shape[0])

        augmented_x = model(x).to("cpu")

        plt.figure(figsize=(8, 8))

        plt.subplot(1, 2, 1)
        original_image = x[0].permute(1, 2, 0).detach().numpy()
        # Normalize to [0, 1] if it's a float image
        if original_image.dtype == 'float32' or original_image.dtype == 'float64':
            original_image = np.clip(original_image, 0, 1)
        plt.imshow(original_image)
        plt.title("Original Image CLass: " + str(labels[0].item()))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        augmented_image = augmented_x[0].permute(1, 2, 0).detach().numpy()
        # Normalize to [0, 1] if it's a float image
        if augmented_image.dtype == 'float32' or augmented_image.dtype == 'float64':
            augmented_image = np.clip(augmented_image, 0, 1)
        plt.imshow(augmented_image)
        plt.title("Augmented")
        plt.axis("off")

        plt.show()
