import os
from model import MainModel
import torch
import torch.nn as nn
from training_utilities import BCEWithLogitsLoss
from hf_dataset_generator import val_transforms, hf_dataset
from datasets import load_dataset
from torch.utils.data import DataLoader


MODEL_DIR = "/home/sahmaran/Desktop"
MODEL_WEIGHTS_NAME = "model_1_cp281.pt"
EMA_WEIGHTS_NAME = "ema_model_cp281.pt"
HF_DIR = "/home/sahmaran/Desktop/IMGNET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODEL = True




def preprocess_weights(weights:dict, 
                       excluded_key:list = "module."):
    new_weights = {}
    for key in weights.keys():
            new_weights[key.replace(excluded_key, "")] = weights[key].to("cpu")
    return new_weights

def return_model(model_path:str = MODEL_DIR, 
                 model_weights_name:str = MODEL_WEIGHTS_NAME,
                 ema_weights_name:str = EMA_WEIGHTS_NAME):
    weights = torch.load(os.path.join(model_path, model_weights_name))
    ema_weights = torch.load(os.path.join(model_path, ema_weights_name))

    model = MainModel.from_dict(**weights["model_config"])
    model.load_state_dict(preprocess_weights(weights["model_state_dict"]))
    model.eval()
    
    ema_model = MainModel.from_dict(**weights["model_config"])
    ema_model.load_state_dict(preprocess_weights(ema_weights))
    ema_model.eval()

    return model, ema_model

def return_dataloader():
        dset_test = load_dataset('imagenet-1k', 
                        keep_in_memory=False,
                        cache_dir = HF_DIR,
                        num_proc = 4, 
                        )["validation"]  
        val_transforms_ = val_transforms(image_size = (320, 320), 
                                         crop_size= (224, 224))
        dset_test =  hf_dataset(dset_test, val_transforms_)
        test_data = DataLoader(dataset= dset_test, batch_size = 256, shuffle = False)
        return test_data



def run_test():
    model, ema_model = return_model()
    model, ema_model = model.to(DEVICE), ema_model.to(DEVICE)
    model.eval()
    ema_model.eval()

    model = torch.compile(model) if COMPILE_MODEL else model
    ema_model = torch.compile(ema_model) if COMPILE_MODEL else ema_model

    test_data = return_dataloader()

    loss_fn_1, loss_fn_2 = nn.CrossEntropyLoss(), BCEWithLogitsLoss(num_classes=1000, label_smoothing=0.0)

    temp_loss_1, temp_loss_2 = 0, 0
    acc = 0
    size = 0
    num_batch = 0

    for images, labels in test_data:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(images)
            temp_loss_1 += loss_fn_1(outputs, labels).item()
            temp_loss_2 += loss_fn_2(outputs, labels).item()
            acc += (outputs.argmax(1) == labels).sum().item()
            size += len(labels)
            num_batch += 1
        print(f"CrossEntropyLoss: {temp_loss_1/num_batch}, BCEWithLogitsLoss: {temp_loss_2/num_batch}, Accuracy: {acc/size}")
    
    

if __name__ == "__main__":
    run_test()
