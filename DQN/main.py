from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

DATA_DIR = Path("./data")
CIFAR_ROOT = DATA_DIR / "cifar-10-batches-py"
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

_datasets = {}


def _load_cifar10(train: bool):
    return torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=train,
        download=not CIFAR_ROOT.exists(),
        transform=transform,
    )


def load_datasets():
    if "train" not in _datasets:
        full_train_dataset = _load_cifar10(train=True)
        test_dataset = _load_cifar10(train=False)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        _datasets["train"] = train_dataset
        _datasets["val"] = val_dataset
        _datasets["test"] = test_dataset

    return _datasets["train"], _datasets["val"], _datasets["test"]


def get_loaders(batch_size=BATCH_SIZE):
    train_dataset, val_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_loaders()

def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

class DQN(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=100, num_hidden_layers=20):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(state_size, hidden_size),
            nn.SiLU()
        ]
        
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU()
            ])
        
        layers.append(nn.Linear(hidden_size, num_actions))

        self.network = nn.Sequential(*layers)
        self.apply(use_he_init)

    def forward(self, state):
        return self.network(state)

model = DQN(state_size=3072, num_actions=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device(
    "mps" if torch.mps.is_available() else "cpu"
)
print(f"Using {device}")

model = model.to(device)

bad_epochs = 10
patience = 10
prev_val_loss = float('inf')
best_model_state = None

num_epochs = 100

for i in range(num_epochs):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss/len(train_loader)

    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss <= prev_val_loss:
            prev_val_loss = avg_val_loss
            bad_epochs = patience
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs -= 1
            if bad_epochs == 0:
                print("Early stopping occurring now")
                break
            else:
                print(f"Bad epoch detected: {bad_epochs} remaining until stopping")

        print(
            f"Epoch [{i + 1}/{num_epochs}] "
            f"train loss: {avg_loss:.4f}, val loss: {avg_val_loss:.4f}"
        )

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()

total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")



