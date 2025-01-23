import os
import random
import time
import copy
from collections import Counter
import numpy as np
import torch
torch.cuda.empty_cache()
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

class CatBreedClassifier:
    def __init__(self, train_dir, test_dir, batch_size=64, lr=0.0006, momentum=0.9, step_size=20, gamma=0.1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self._set_random_seed(2025)
        self._prepare_data()
        self._initialize_model()

    def _set_random_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_data(self):
        # Transformation
        train_transform = ResNet50_Weights.DEFAULT.transforms()
        test_transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self.train_data = ImageFolder(self.train_dir, train_transform)
        self.test_data = ImageFolder(self.test_dir, test_transform)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        self.dataloaders = {"train": self.train_loader, "test": self.test_loader}
        self.dataset_sizes = {"train": len(self.train_data), "test": len(self.test_data)}
        self.classes = sorted(os.listdir(self.train_dir))
        self.index_to_class = {v: k for k, v in self.test_data.class_to_idx.items()}

    def _initialize_model(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def train(self, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(self.dataloaders[phase], desc=phase, position=0, leave=True):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {best_acc:4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        # Save the best model if required
        torch.save(self.model, "app/model.pt")
        print("Best model saved as 'app/model.pt'")

    def evaluate(self):
        running_corrects = 0
        pred = []
        target = []

        for inputs, labels in tqdm(self.dataloaders["test"], desc="test", position=0, leave=True):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            target += labels.cpu().detach().numpy().tolist()

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                pred += preds.cpu().detach().numpy().tolist()

            running_corrects += torch.sum(preds == labels.data)

        correct_counts = Counter()
        total_counts = Counter(target)

        for t, p in zip(target, pred):
            if t == p:
                correct_counts[t] += 1

        accuracy = {}
        for c in range(len(self.classes)):
            total = total_counts[c]
            accuracy[c] = correct_counts[c] / total if total > 0 else None

        for c in range(len(self.classes)):
            class_name = self.classes[c]
            acc = accuracy[c]
            if acc is not None:
                print(f"Class {class_name} (ID: {c}) - Accuracy: {acc:.2%}")
            else:
                print(f"Class {class_name} (ID: {c}) - No samples in test set")
                
                


if __name__ == "__main__":
    TRAIN_DIR = r"C:\\Users\\Terence\\Desktop\\35-cat-breed-dataset\\train"
    TEST_DIR = r"C:\\Users\\Terence\\Desktop\\35-cat-breed-dataset\\test"

    classifier = CatBreedClassifier(TRAIN_DIR, TEST_DIR)
    classifier.train(num_epochs=30)
    classifier.evaluate()