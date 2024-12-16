import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Define the CNN model (same as before)
from model import MainModel
from tqdm import tqdm
import numpy as np

### This model is easy peasy for the sake of testing, it is not a good model for CIFAR100

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.device_count()>=1 else "cpu")
torch.set_float32_matmul_precision('high')

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
cifar10_mean =  [0.4914, 0.4822, 0.4465]
cifar10_std =  [0.2470, 0.2435, 0.2616]
NUM_CLASSES = 10



def return_data():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RGB(),
        transforms.RandomResizedCrop(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale = True),
        transforms.Normalize(cifar10_mean, cifar100_std),
        transforms.RandomErasing(0.25)

    ])
    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale = True),
        transforms.Normalize(cifar10_mean, cifar100_std)
    ])

    # Assuming cutmix_or_mixup is defined globally
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=0.8)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    # Define the collate function at the top level  
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transform)
    trainloader = DataLoader(trainset, batch_size = 256, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size = 256, shuffle=False)
    return trainloader, testloader


def return_model(compiled = False):

    torch.manual_seed(0)
    model = MainModel(
        embedding_dim=256,
        num_blocks=4,
        n_head=8,
        activation="gelu",
        conv_kernel_size=5,
        patch_size=2,
        ffn_dropout=0.2,
        attn_dropout=0.2,
        output_classes=NUM_CLASSES,
        conv_block_num=2,
        ff_multiplication_factor=4,
        max_image_size=[16, 16],
        max_num_registers=5,
        embedding_activation="none",
        conv_first=False,
        head_output_from_register=False,
        simple_mlp_output=False,
        output_head_bias=False,
        normalize_qv=True,
        stochastic_depth_p=[0.0, 0.0],
        mixer_deptwise_bias=False,
        mixer_ffn_bias=False,
        fast_att=True,
    )
    if compiled and device !="mps":
        model = torch.compile(model.to(device), fullgraph = True, dynamic=True)
        Warning("Model is can not be compiled when model is on MPS device")
    else:
        model = model.to(device)
    from training_utilities import BCEWithLogitsLoss
    loss_fn = BCEWithLogitsLoss(num_classes=NUM_CLASSES, label_smoothing=0.0)
    from torch_optimizer import Lamb 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ## ema model
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.90))
    ema_model = ema_model.to(device)

    scheduler0 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.001)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 7 , 150)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler0, scheduler1, scheduler2], milestones= [1, 5])
    return model, loss_fn, optimizer, ema_model, scheduler

from training_tools import EMA_model

# Training loop
def train(epochs, 
          model, 
          ema_model,
          trainloader, 
          testloader,
          loss_fn,
          optimizer,
          scheduler):
    
    ema = EMA_model(model, 0.95)


    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(trainloader)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += 1
            pbar.set_postfix({'loss': loss.item(), "lr": optimizer.param_groups[0]["lr"]})
 

            ema_model.update_parameters(model)
            ema.update_parameters(model)

        scheduler.step()
        for key, value in model.state_dict().items():
            if "running" in key:
                print(key, value.mean().item())

        epoch_loss = running_loss / total
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')


        # Evaluate on test set
        model_test_acc = evaluate(testloader, model)
        ema_model_test_acc = evaluate(testloader, ema_model)
        print(f'Test Accuracy: {model_test_acc:.2f}%, EMA Test Accuracy: {ema_model_test_acc:.2f}%')

# Evaluation function (same as before)
def evaluate(testloader, model):
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
    model, loss_fn, optimizer, ema_model, scheduler = return_model()
    print(f"This model has {model.return_num_params()} parameters")
    train(epochs=150, model = model, ema_model= ema_model, trainloader=trainloader, testloader=testloader, loss_fn=loss_fn, optimizer=optimizer, scheduler = scheduler)
