# modified from https://github.com/facebookresearch/DiT/blob/main/download.py
from torchvision.datasets.utils import download_file_from_google_drive
import torch
import os

GGDRIVE_ID = {
    "MSS_resnet50.pt": "1T2dNN931VnN1EUn8b3lmZwaY3mVxHYWg",
    "TM_resnet50.pt": "1NL-rvvusxhbVCg4GkPbWANelpyIk_QF_",
    "OLITIRS_resnet50.pt": "1smOhaM635ilQMOsFjlV5d-mLRBKJzfot",
}

PRETRAINED_MODELS = {
    "MSS": "MSS_resnet50.pt",
    "TM": "TM_resnet50.pt",
    "OLITIRS": "OLITIRS_resnet50.pt",
}

def download_model(sensor_type):
    """
    Downloads a pre-trained ResNet-50 model from Google Drive.
    """
    assert sensor_type in PRETRAINED_MODELS.keys()

    model_name = PRETRAINED_MODELS[sensor_type]
    root = "pretrained_models"
    local_path = os.path.join(root, model_name)
    
    if not os.path.isfile(local_path):
        id = GGDRIVE_ID[model_name]
        os.makedirs(root, exist_ok=True)
        download_file_from_google_drive(id, root, model_name)
    
    model = torch.load(local_path, map_location=torch.device('cpu'))
    return model