import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import EMNISTClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_ROOT     = "./data"
SAVE_PATH     = "emnist_model.pth"
BATCH_SIZE    = 64
TEST_BATCH    = 1000
EPOCHS        = 10
LEARNING_RATE = 0.001
LOG_INTERVAL  = 1000
MEAN, STD     = 0.1751, 0.3332

# ── Data ───────────────────────────────────────────────────────────────────────
def fix_emnist_orientation(tensor):
    """EMNIST images are rotated 90° and mirrored, this corrects that."""
    return tensor.transpose(1, 2)

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Lambda(fix_emnist_orientation),
    transforms.Normalize((MEAN,), (STD,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(fix_emnist_orientation),
    transforms.Normalize((MEAN,), (STD,))
])

def get_dataloaders():
    train_dataset = datasets.EMNIST(root=DATA_ROOT, split='balanced', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.EMNIST(root=DATA_ROOT, split='balanced', train=False, download=True, transform=test_transform)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=TEST_BATCH,  shuffle=False)
    return train_loader, test_loader

# ── Training ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loss_fn, optimizer, loader, device):
    """Runs one full training epoch. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss, interval_loss = 0.0, 0.0
    correct, count = 0, 0

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted   = outputs.max(1)
        total_loss    += loss.item()
        interval_loss += loss.item()
        count   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % LOG_INTERVAL == 0:
            print(f"  Step [{i+1:>4}]  Loss: {interval_loss / LOG_INTERVAL:.4f}  |  Acc: {100 * correct / count:.2f}%")
            interval_loss = 0.0

    return total_loss / len(loader), 100 * correct / count


def evaluate(model, loader, device):
    """Evaluates model on loader. Returns accuracy %."""
    model.eval()
    correct, count = 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted   = model(inputs).max(1)
            count   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct / count


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    train_loader, test_loader = get_dataloaders()

    model     = EMNISTClassifier().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, loss_fn, optimizer, train_loader, device)
        test_acc              = evaluate(model, test_loader, device)
        print(f"  → Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%  |  Test Acc: {test_acc:.2f}%\n")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved → {SAVE_PATH}")