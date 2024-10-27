import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from ProgRandConv import ProgRandConvBlock
from copy import deepcopy
import loaders

class Model:
    def __init__(
        self,
        domain,
        config
    ):
        """Consists of a network and an optimizer and their configuration"""
        self.domain = domain
        self.config = config
        self.device = config["device"]
        
        self._init_model()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["alpha"], momentum=config["momentum"]
        )

        # cosine learning schedule for self.schedule_epochs epochs
        if config["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["Tmax"])

        self.criterion = nn.CrossEntropyLoss()

        self.train_loader, self.test_loader = self._load_dataset(self.domain, config["batch_size"])

    def _load_dataset(self, domain, batch_size):
        print(domain)
        if domain == "digits":
            # Trained on MNIST and tested on other Digit datasets
            train_loader, test_loader = loaders.MNISTLoader(root="data/", batch_size=batch_size, num_workers=4).get_loaders()
        elif domain == "pacs":
            train_dataset = loaders.PACSDataset(root_dir="data/archive", domain=self.config["source"], train=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            test_dataset = loaders.PACSDataset(root_dir="data/archive", domain=self.config["source"], train=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        elif domain == "office":
            pass
        else:
            raise ValueError("Dataset not found. Available datasets: digits, pacs, office")
        return train_loader, test_loader

    def _init_scheduler(self):
        if self.config["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config["Tmax"])
        else:
            self.scheduler = None
            raise ValueError("Scheduler not found. Available schedulers: cosine")

    def _init_model(self):
        model_name = self.config["arch"]    

        if model_name == "LeNet":
            self.model = LeNet().to(self.device)
        elif model_name == "AlexNet":
            self.model = AlexNet().to(self.device)
        elif model_name == "ResNet18":
            self.model = ResNet18().to(self.device)
        elif model_name == "ResNet50":
            self.model = ResNet50().to(self.device)
        else:
            raise ValueError("Model not found. Available models: LeNet, AlexNet, ResNet18, ResNet50")


    def train_with_progressive_augmentation(self):
        num_epochs = self.config["Tmax"]
        best_val_acc = 0
        for t in range(num_epochs):

            losses = 0
            iter = 0
            for i, (X0, y) in enumerate(self.train_loader):
                X0, y = X0.to(self.device), y.to(self.device)
                if X0.shape[1] == 1:
                    X0 = X0.repeat(1, 3, 1, 1).to(self.device)
                # Initialize random convolution block parameters
                prog_rand_conv = ProgRandConvBlock(in_channels=X0.shape[1], out_channels=3, kernel_size=3, l_max=self.config["Lmax"], device=self.config["device"], batch_size=X0.shape[0], sigma_d = self.config["sigma_d"], input_dim=X0.shape[-1])
                augmented_x = prog_rand_conv(X0)
                pred = self.model(augmented_x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss.item()

                if iter % 100 == 0 and iter != 0:
                    print(f"Epoch {t+1} / {num_epochs}, Step {i+1} / {len(self.train_loader)}, Loss: {losses/iter}")
                iter += 1

            print(f"Epoch {t+1}, Loss: {losses/len(self.train_loader)}")

            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                total = 0
                correct = 0
                for i, (X0, y) in enumerate(self.test_loader):
                    X0, y = X0.to(self.device), y.to(self.device)  # Test verilerini de CUDA'ya taşıdık
                    if X0.shape[1] == 1:
                        X0 = X0.repeat(1, 3, 1, 1).to(self.device)
                    self.model.eval()
                    outputs = self.model(X0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            if correct / total > best_val_acc:
                torch.save(self.model.state_dict(), f'{self.domain}_{self.config["source"]}_best_model_weights.pth') 
            print(f"Accuracy: {100 * correct / total}%")

        torch.save(self.model.state_dict(), f'{self.domain}_{self.config["source"]}_final_model_weights.pth') 


class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)  # Adjusted to 16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)  # Adjusted to 16*5*5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(weights=None)
        self.alexnet.features[0] = nn.Conv2d(
            input_channels, 64, kernel_size=11, stride=4, padding=2
        )
        self.alexnet.classifier[6] = nn.Linear(
            4096, num_classes
        )  # Change the output layer to 10 classes

    def forward(self, x):
        return self.alexnet(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(
            512, num_classes
        )  # Change the output layer to 10 classes

    def forward(self, x):
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(
            2048, num_classes
        )  # Change the output layer to 10 classes

    def forward(self, x):
        return self.resnet(x)
