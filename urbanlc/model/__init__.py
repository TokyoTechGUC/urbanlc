from .base import *
from .baseline import *
from .dataloader import *
from .download import *
from .train_utils import *
from .pipeline_transforms import *
from .deep_learning import (
    DeepLearningLCC,
    MSSDeepLearning,
    TMDeepLearning,
    OLITIRSDeepLearning,
)
from typing import Optional, Union, Dict, Tuple, Any
from torchvision.datasets.utils import download_file_from_google_drive
import os
import yaml

MODEL_GGDRIVE_ID = {
    "MSS_resnet50.pt": "1T2dNN931VnN1EUn8b3lmZwaY3mVxHYWg",
    "TM_resnet50.pt": "1NL-rvvusxhbVCg4GkPbWANelpyIk_QF_",
    "OLITIRS_resnet50.pt": "1smOhaM635ilQMOsFjlV5d-mLRBKJzfot",
}

CONFIG_GGDRIVE_ID = {
    "MSS_resnet50.yml": "1BlAaAkS4SwNA3IA_WvK-vJ1uFVNIKAes",#https://drive.google.com/file/d/1BlAaAkS4SwNA3IA_WvK-vJ1uFVNIKAes/view?usp=drive_link
    "TM_resnet50.yml": "1iCpEY3wA7_jLmw-QffsadOjzcxSEc1hZ",#https://drive.google.com/file/d/1iCpEY3wA7_jLmw-QffsadOjzcxSEc1hZ/view?usp=drive_link
    "OLITIRS_resnet50.yml": "1qygGBmZphBhh3m0ZsG4DAh_QKyC898jH",#https://drive.google.com/file/d/1qygGBmZphBhh3m0ZsG4DAh_QKyC898jH/view?usp=drive_link
}

PRETRAINED_MODELS = ["MSS_resnet50", "TM_resnet50", "OLITIRS_resnet50"]

def download_model_and_config(model_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Downloads a pre-trained model and its configuration from Google Drive.

    :param model_id: Identifier for the pre-trained model.
    :type model_id: str
    :return: Tuple containing the model path and model parameters.
    :rtype: Tuple[str, Dict[str, Any]]
    """
    assert model_id in PRETRAINED_MODELS

    model_name = f"{model_id}.pt"
    config_name = f"{model_id}.yml"
    root = "pretrained_models"

    # download model
    model_path = os.path.join(root, model_name)
    if not os.path.isfile(model_path):
        os.makedirs(root, exist_ok=True)
        id = MODEL_GGDRIVE_ID[model_name]
        download_file_from_google_drive(id, root, model_name)

    # download config file
    config_path = os.path.join(root, config_name)
    if not os.path.isfile(config_path):
        os.makedirs(root, exist_ok=True)
        id = CONFIG_GGDRIVE_ID[config_name]
        download_file_from_google_drive(id, root, config_name)

    # extract necessary parameters from config file
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    params = {
        "architecture": configs["architecture"],
        "model_params": configs["model_params"],
        "device": "cpu",
        "save_path": configs["save_path"],
    }

    return model_path, params

class LCClassifier(DeepLearningLCC):
    """
    Land Cover Classification (LCC) model
    """
    
    @classmethod
    def from_pretrained(
        cls,
        sensor: str,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        model_params: Optional[Dict[str, str]] = None,
    ):
        """
        Create an instance of LCC classifier from a pre-trained model.

        :param sensor: The type of Landsat sensor (MSS, TM, OLITIRS).
        :type sensor: str
        :param pretrained_model_name_or_path: Name or path of the pre-trained model.
        :type pretrained_model_name_or_path: Union[str, os.PathLike]
        :param model_params: Optional dictionary of model parameters.
        :type model_params: Optional[Dict[str, str]]
        :return: LCClassifier instance.
        :rtype: LCClassifier
        """
        assert sensor in ["MSS", "TM", "OLITIRS"]

        if pretrained_model_name_or_path in PRETRAINED_MODELS:
            print("Initialize using pretrained weights")
            checkpoint_path, model_params = download_model_and_config(pretrained_model_name_or_path)
            model = globals()[f"{sensor}DeepLearning"](**model_params)
            model.load_model(checkpoint_path)
        else:
            model = globals()[f"{sensor}DeepLearning"](**pretrained_model_name_or_path[0])
            model.load_model(pretrained_model_name_or_path[1])

        return model