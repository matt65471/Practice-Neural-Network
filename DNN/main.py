from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

DATA_DIR = Path("./data")
CIFAR_ROOT = DATA_DIR / "cifar-10-batches-py"
BATCH_SIZE = 64
RUN_NAME = "cifar_dnn_mc_dropout"
VAL_LOSS_LOG_PATH = Path(__file__).parent / "val_loss_history.csv"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{RUN_NAME}.pt"
ALPHA_DROPOUT_P = 0.1
MC_DROPOUT_SAMPLES = 30
SKIP_TRAINING = CHECKPOINT_PATH.exists()

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


def log_val_loss(run_name: str, epoch: int, val_loss: float, log_path: Path = VAL_LOSS_LOG_PATH) -> None:
    if not log_path.exists():
        log_path.write_text("run_name,epoch,val_loss\n")

    with log_path.open("a") as f:
        f.write(f"{run_name},{epoch},{val_loss:.6f}\n")


train_loader, val_loader, test_loader = get_loaders()

def use_lecun_init(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, math.sqrt(1 / module.in_features))
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class DNN(nn.Module):
    def __init__(
        self,
        state_size,
        num_actions,
        hidden_size=100,
        num_hidden_layers=20,
        dropout_p=ALPHA_DROPOUT_P,
    ):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(state_size, hidden_size),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_p),
        ]

        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SELU(),
                nn.AlphaDropout(p=dropout_p),
            ])

        layers.append(nn.Linear(hidden_size, num_actions))

        self.network = nn.Sequential(*layers)
        self.apply(use_lecun_init)

    def forward(self, state):
        return self.network(state)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def evaluate_mc_dropout(model, loader, device, n_samples=MC_DROPOUT_SAMPLES):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.AlphaDropout)):
            module.train()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            avg_probs = None

            for _ in range(n_samples):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                avg_probs = probs if avg_probs is None else avg_probs + probs

            predicted = (avg_probs / n_samples).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.eval()
    return correct / total


model = DNN(state_size=3072, num_actions=10)
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
val_losses = []

if SKIP_TRAINING:
    print(f"Loading checkpoint from {CHECKPOINT_PATH} (skipping training)")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
else:
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
            val_losses.append(avg_val_loss)
            log_val_loss(RUN_NAME, i + 1, avg_val_loss)

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

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / RUN_NAME)

    if best_model_state is not None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, CHECKPOINT_PATH)
        model.load_state_dict(best_model_state)

standard_accuracy = evaluate(model, test_loader, device)
mc_accuracy = evaluate_mc_dropout(model, test_loader, device)

print(f"Standard test accuracy: {standard_accuracy:.2%}")
print(f"MC dropout test accuracy ({MC_DROPOUT_SAMPLES} samples): {mc_accuracy:.2%}")
print(f"Change: {(mc_accuracy - standard_accuracy) * 100:+.2f} percentage points")

plt.show()