import random
import numpy as np
import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

def set_seed(seed):
    """
    Set random seeds for reproducibility in random, numpy, and PyTorch.

    This function sets random seeds for the 'random', 'numpy', and 'torch' modules,
    and ensures deterministic behavior on GPU.

    :param seed: Seed value for random number generation.
    :type seed: int

    :return: None
    """

    # for some reasons, setting torch.backends.cudnn.deterministic = True will return an error
    # Can't pickle CudnnModule objects
    # ref: https://github.com/ray-project/ray/issues/8569
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    eval('setattr(torch.backends.cudnn, "benchmark", True)')
    eval('setattr(torch.backends.cudnn, "deterministic", True)')


def save_checkpoint(
    save_dir: str,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
) -> str:
    """
    Save model checkpoint, including model state, optimizer state, scheduler state, and epoch.
    
    This function saves a model checkpoint to a file in the specified directory.
    The checkpoint includes the model state, optimizer state, scheduler state, and current epoch.
    
    :param save_dir: Directory to save the checkpoint file.
    :type save_dir: str
    :param name: Name of the checkpoint file.
    :type name: str
    :param model: PyTorch model to be saved.
    :type model: nn.Module
    :param optimizer: Optimizer used during training.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Learning rate scheduler. Defaults to None.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param epoch: Current epoch.
    :type epoch: int

    :return: Filepath of the saved checkpoint.
    :rtype: str
    """
    temp_model = copy.deepcopy(model)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{name}.pt")
    state = {
        "model": temp_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, filepath)
    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]:
    """
    Load model checkpoint, including model state, optimizer state, scheduler state, and epoch.

    This function loads a model checkpoint from a file and returns the loaded model, optimizer, scheduler, and epoch.

    :param checkpoint_path: Path to the checkpoint file.
    :type checkpoint_path: str
    :param model: PyTorch model to be loaded.
    :type model: nn.Module
    :param optimizer: Optimizer to be loaded. Defaults to None.
    :type optimizer: Optional[torch.optim.Optimizer]
    :param scheduler: Learning rate scheduler to be loaded. Defaults to None.
    :type scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    :param device: Device to load the checkpoint. Defaults to GPU if available.
    :type device: torch.device

    :return: Loaded model, optimizer, scheduler, and epoch.
    :rtype: Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = checkpoint["model"]

    # Check if the model_dict key matches that of the model itself
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    elapsed_epoch = checkpoint.get("epoch", 0)

    return model, optimizer, scheduler, elapsed_epoch


def segment_satelite_image(
    img: torch.Tensor, sub_size: Optional[int] = 224, stride: Optional[int] = None
) -> Tuple[List[torch.Tensor], List[tuple]]:
    """
    Partition a satellite image into patches of specified size and stride.

    This function takes a satellite image and partitions it into patches of a specified size and stride.
    It returns a list of partitioned patches and a list of corresponding coordinates (bounding boxes) for each patch.

    :param img: Input satellite image.
    :type img: torch.Tensor
    :param sub_size: Size of the patches. Defaults to 224.
    :type sub_size: Optional[int]
    :param stride: Stride for patch extraction. Defaults to None.
    :type stride: Optional[int]

    :return: List of partitioned patches and corresponding coordinates.
    :rtype: Tuple[List[torch.Tensor], List[tuple]]
    """
    assert isinstance(sub_size, int)
    if img.dim() == 3:
        if isinstance(img, torch.Tensor):
            img = torch.unsqueeze(img, dim=0)
        elif isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0)
        else:
            raise TypeError("Image must be a NumPy array or a Torch tensor")

    stride = sub_size if stride is None else stride
    data_out = []
    coordinate_out = []
    remain_w = img.shape[2] % stride
    remain_h = img.shape[3] % stride
    for i in range(0, img.shape[2] - sub_size, stride):
        for j in range(0, img.shape[3] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

    if remain_h or remain_w:
        # remain_width
        i = img.shape[2] - sub_size
        for j in range(0, img.shape[3] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

        # remain_height
        j = img.shape[3] - sub_size
        for i in range(0, img.shape[2] - sub_size, stride):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i + sub_size), int(j + sub_size)
            data_out.append(img[:, :, min_w:max_w, min_h:max_h])
            coordinate_out.append((min_w, max_w, min_h, max_h))

        # the corner patch
        min_w, min_h = img.shape[2] - sub_size, img.shape[3] - sub_size
        max_w, max_h = img.shape[2], img.shape[3]
        data_out.append(img[:, :, min_w:max_w, min_h:max_h])
        coordinate_out.append((min_w, max_w, min_h, max_h))

    return data_out, coordinate_out


# naive implementation
# TODO: optimize this
def combine_prediction(
    preds: torch.Tensor,
    coordinates: List[Tuple[int]],
    original_size: Tuple[int],
    method: Optional[str] = "mean",
) -> torch.Tensor:
    """
    Combine predictions from partitioned patches into a complete prediction.

    This function combines predictions from partitioned patches into a complete prediction for the entire image.
    The method for combining predictions in the overlapped regions can be specified (default is mean).

    :param preds: Predictions from partitioned patches.
    :type preds: torch.Tensor
    :param coordinates: List of coordinates (bounding boxes) for each patch.
    :type coordinates: List[Tuple[int]]
    :param original_size: Original size of the complete prediction.
    :type original_size: Tuple[int]
    :param method: Method for combining predictions in the overlapped regions. Defaults to "mean".
    :type method: Optional[str]

    :return: Combined prediction.
    :rtype: torch.Tensor
    """
    if method == "mean":
        # calculate mean probability of each pixel from patches and select the highest one
        # reduce memory consumption
        softmax = nn.Softmax(dim=0)
        combined_preds = []
        num_patches = preds.shape[0]
        for element in preds:
            output = torch.zeros(11, original_size[0], original_size[1])
            # actually the logic for calculating mean prob is incorrect, but the order is still preserved anyway
            for _, (patch, bounds) in enumerate(zip(element, coordinates)):
                min_w, max_w, min_h, max_h = bounds
                output[:, min_w:max_w, min_h:max_h] = output[:, min_w:max_w, min_h:max_h] + (1.0 / num_patches) * (
                    softmax(patch) - output[:, min_w:max_w, min_h:max_h]
                )
            combined_preds.append(output)

        combined_preds = torch.stack(combined_preds, axis=0)
        return combined_preds
    else:
        raise NotImplementedError