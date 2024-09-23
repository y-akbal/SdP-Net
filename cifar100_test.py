import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import autoaugment, transforms
# Define the CNN model (same as before)
from model import MainModel
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.device_count()>=1 else "cpu")
torch.set_float32_matmul_precision('high')

def return_data():

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

    # Assuming cutmix_or_mixup is defined globally
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=0.8)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    # Define the collate function at the top level  
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform = test_transform)

    trainloader = DataLoader(trainset, batch_size = 128, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size = 64, shuffle=False)

    return trainloader, testloader


def return_model(compiled = False):

    torch.manual_seed(0)
    model = MainModel(
        n_head = 8, 
        activation = nn.GELU(),
        patch_size = 8, 
        conv_kernel_size = 7, 
        output_classes = 100, 
        embedding_dim = 128, 
        num_blocks = 10,
        max_image_size = (16,16),
        head_output_from_register = False,
        simple_mlp_output = True,
        stochastic_depth = True,
        stochastic_depth_p = [0.9, 0.001],
        normalize_qv = True
    )
    if compiled and device !="mps":
        model = torch.compile(model.to(device))
        Warning("Model is can not be compiled when model is on MPS device")
    else:
        model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, loss_fn, optimizer

# Training loop
def train(epochs, 
          trainloader, 
          testloader,
          loss_fn,
          optimizer):
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(trainloader)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += 1
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / total
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

        # Evaluate on test set
        test_acc = evaluate(testloader)
        print(f'Test Accuracy: {test_acc:.2f}%')

# Evaluation function (same as before)
def evaluate(testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = outputs.argmax(-1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    trainloader, testloader = return_data()
    model, loss_fn, optimizer = return_model()
    print(f"This model has {model.return_num_params()[0]} parameters")
    train(epochs=100, trainloader=trainloader, testloader=testloader, loss_fn=loss_fn, optimizer=optimizer)
    
