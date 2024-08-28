import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.data import default_collate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import autoaugment, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
train_transform = transforms.Compose([
    autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])



NUM_CLASSES = 100

cutmix = v2.CutMix(num_classes = NUM_CLASSES)
mixup = v2.MixUp(num_classes = NUM_CLASSES, alpha = 0.8)

cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
collate_fn = lambda batch : cutmix_or_mixup(*default_collate(batch))





# Load CIFAR100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16, collate_fn=collate_fn)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Define the CNN model (same as before)
from model import main_model
import torch
from torch import nn
model = main_model(
    n_head= 4, 
    activation= nn.GELU("tanh"),
    patch_size= 4, 
    conv_kernel_size=5, 
    output_classes=100, 
    embedding_dim=128, 
    num_blocks=5,
    max_image_size=(16,16),
    head_output_from_register=False,
    stochastic_depth=True,
    stochastic_depth_p=[0.3, 0.001]
)


# Initialize the model, loss function, and optimizer
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[0], labels)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs[0].data, 1)
            total += inputs.size(0)

        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        test_acc = evaluate()
        print(f'Test Accuracy: {test_acc:.2f}%')

# Evaluation function (same as before)
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Run training
train(epochs=100)