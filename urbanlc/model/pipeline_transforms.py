import rasterio
import numpy as np
import random
import os
import glob
from natsort import natsorted
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from .train_utils import set_seed
import warnings

try:
    from torchgeo.transforms import indices
except Exception:
    warnings.warn("torchgeo is not installed, data augmentation is disabled.")

##############################
# XGBoost
##############################

import numpy as np

def compute_NDVI(img: np.ndarray, index_nir: int, index_red: int) -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    :param img: Landsat image.
    :type img: np.ndarray
    :param index_nir: Index of the Near-Infrared (NIR) band in the image.
    :type index_nir: int
    :param index_red: Index of the Red band in the image.
    :type index_red: int
    :return: NDVI computed from the input image.
    :rtype: np.ndarray
    """
    nir = img[index_nir, :, :]
    red = img[index_red, :, :]

    out = (nir - red) / (nir + red)
    out[np.isnan(out)] = -1.0
    return np.expand_dims(out, axis=0)

def compute_NDBI(img: np.ndarray, index_swir: int, index_nir: int) -> np.ndarray:
    """
    Compute the Normalized Difference Built-Up Index (NDBI).

    :param img: Landsat image.
    :type img: np.ndarray
    :param index_swir: Index of the Shortwave Infrared (SWIR) band in the image.
    :type index_swir: int
    :param index_nir: Index of the Near-Infrared (NIR) band in the image.
    :type index_nir: int
    :return: NDBI computed from the input image.
    :rtype: np.ndarray
    """
    nir = img[index_nir, :, :]
    swir = img[index_swir, :, :]

    out = (swir - nir) / (swir + nir)
    out[np.isnan(out)] = -1.0
    return np.expand_dims(out, axis=0)

def compute_BUI(img: np.ndarray, index_a: int, index_b: int) -> np.ndarray:
    """
    Compute the Built-Up Index (BUI).

    :param img: Landsat image.
    :type img: np.ndarray
    :param index_a: Index of a band used in the computation of the BUI.
    :type index_a: int
    :param index_b: Index of another band used in the computation of the BUI.
    :type index_b: int
    :return: BUI computed from the input image.
    :rtype: np.ndarray
    """
    ndbi = img[index_a, :, :]
    ndvi = img[index_b, :, :]

    out = ndbi - ndvi
    out[np.isnan(out)] = -1.0
    return np.expand_dims(out, axis=0)

def compute_NDWI(img: np.ndarray, index_green: int, index_nir: int) -> np.ndarray:
    """
    Compute the Normalized Difference Water Index (NDWI).

    :param img: Landsat image.
    :type img: np.ndarray
    :param index_green: Index of the Green band in the image.
    :type index_green: int
    :param index_nir: Index of the Near-Infrared (NIR) band in the image.
    :type index_nir: int
    :return: NDWI computed from the input image.
    :rtype: np.ndarray
    """
    nir = img[index_nir, :, :]
    green = img[index_green, :, :]

    out = (green - nir) / (green + nir)
    out[np.isnan(out)] = -1.0
    return np.expand_dims(out, axis=0)



##############################
# DEEP LEARNING
##############################

# for landsat8

MEAN_OLITIRS = [
    0.05244060772832586,
    0.06419783589372872,
    0.09355600224058667,
    0.09683895508641474,
    0.20228001018199623,
    0.17475386334671822,
    0.1299458739828,
    295.2477865092859,
    0.0,  # NDBI
    0.0,  # NDVI
    0.0,  # BUI
]

STD_OLITIRS = [
    0.03056170518168943,
    0.034643695523177205,
    0.042781807483223834,
    0.05637172934273404,
    0.09086194854765857,
    0.08659663887712442,
    0.0762477946322169,
    49.351306701659716,
    1.0,  # NDBI
    1.0,  # NDVI
    1.0,  # BUI
]

# for landsat7
MEAN_TM = [
    0.08163042661356694,
    0.10085783731931074,
    0.10701118915780652,
    0.20055640287459503,
    0.1701256492465244,
    0.12407282351539259,
    291.8235398823627,
    0.0,  # NDBI
    0.0,  # NDVI
    0.0,  # BUI
]

STD_TM = [
    0.05398084773803982,
    0.05849992704495288,
    0.06925856941611415,
    0.08748855275238117,
    0.08646376806562776,
    0.07575478301131,
    48.37941596949521,
    1.0,  # NDBI
    1.0,  # NDVI
    1.0,  # BUI
]

MEAN_MSS = [
    58.52628466445192,
    63.99713552734609,
    73.72406260655546,
    67.19410871327327,
    0.0,  # NDVI
]

STD_MSS = [
    20.506525138881113,
    32.3602409954395,
    33.81779753120669,
    30.294028783169907,
    1.0,  # NDVI
]

class AppendBUI(indices.AppendNormalizedDifferenceIndex):
    """
    Compute the Built-Up Index (BUI) for PyTorch Tensor.
    """
    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, int],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Built-Up Index (BUI) for PyTorch Tensor.

        :param input: The input tensor.
        :type input: torch.Tensor
        :param params: Generated parameters.
        :type params: Dict[str, torch.Tensor]
        :param flags: Static parameters.
        :type flags: Dict[str, int]
        :param transform: The geometric transformation tensor.
        :type transform: Optional[torch.Tensor]
        :return: The augmented input.
        :rtype: torch.Tensor
        """
        ndbi = input[..., flags["index_a"], :, :]
        ndvi = input[..., flags["index_b"], :, :]
        bui = ndbi - ndvi
        bui = torch.unsqueeze(bui, -3)
        input = torch.cat((input, bui), dim=-3)
        return input

def rand_bbox(size: Tuple[int], lam: float) -> Tuple[float, float, float, float]:
    """
    Generate a random bounding box.

    :param size: Size of the tensor (N, C, H, W).
    :type size: Tuple[int]
    :param lam: Lambda value for beta distribution.
    :type lam: float
    :return: Coordinates of the bounding box (bbx1, bby1, bbx2, bby2).
    :rtype: Tuple[float, float, float, float]
    """

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_patch(
    image: torch.Tensor,
    gt: torch.Tensor,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix patches in the input image and ground truth following Beta distribution.

    :param image: Input image tensor.
    :type image: torch.Tensor
    :param gt: Ground truth tensor.
    :type gt: torch.Tensor
    :param alpha: Alpha parameter for beta distribution (default: 1.0).
    :type alpha: Optional[float]
    :param beta: Beta parameter for beta distribution (default: 1.0).
    :type beta: Optional[float]
    :return: Mixed image and ground truth tensors.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    lam = np.random.beta(alpha, beta)
    rand_index = torch.randperm(image.size()[0], device=image.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
    gt[:, :, bbx1:bbx2, bby1:bby2] = gt[rand_index, :, bbx1:bbx2, bby1:bby2]

    return image, gt

class LandsatTransformer:
    """
    Input Preprocessor for Landsat data
    """

    def __init__(self, means: List[float], stds: List[float],
                 ndbi_indices: Optional[Dict[str, Any]] = None,
                 ndvi_indices: Optional[Dict[str, Any]] = None,
                 bui_indices: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize input preprocessor for Landsat data.

        :param means: List of mean values for normalization.
        :type means: List[float]
        :param stds: List of standard deviation values for normalization.
        :type stds: List[float]
        :param ndbi_indices: Parameters for NDBI index calculation.
        :type ndbi_indices: Optional[Dict[str, Any]]
        :param ndvi_indices: Parameters for NDVI index calculation.
        :type ndvi_indices: Optional[Dict[str, Any]]
        :param bui_indices: Parameters for BUI index calculation.
        :type bui_indices: Optional[Dict[str, Any]]
        """
        set_seed(0)
        self.means = means
        self.stds = stds
        self.ndbi_indices = ndbi_indices
        self.ndvi_indices = ndvi_indices
        self.bui_indices = bui_indices

        # construct preprocess pipeline
        ops = []
        if self.ndbi_indices is not None:
            ops.append(indices.AppendNDBI(**self.ndbi_indices))
        if self.ndvi_indices is not None:
            ops.append(indices.AppendNDVI(**self.ndvi_indices))
        if (self.bui_indices is not None) and len(ops) == 2:
            ops.append(AppendBUI(**self.bui_indices))

        ops.append(transforms.Normalize(mean=self.means, std=self.stds))
        self.transforms = nn.Sequential(*ops)

    # may running out of memory
    # TODO: implement running statistics
    @staticmethod
    def calculate_statistics(
        root: Optional[str] = None,
        filename_glob: Optional[str] = None,
        all_paths: Optional[List[str]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        num_bands: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate mean and standard deviation statistics of the dataset.

        :param root: Root directory of the dataset.
        :type root: Optional[str]
        :param filename_glob: Glob pattern for selecting files.
        :type filename_glob: Optional[str]
        :param all_paths: List of file paths.
        :type all_paths: Optional[List[str]]
        :param exclude: List of keywords to exclude from file paths.
        :type exclude: Optional[Union[str, List[str]]]
        :param num_bands: Number of bands in the dataset.
        :type num_bands: Optional[int]
        :return: Mean and standard deviation values.
        :rtype: Tuple[List[float], List[float]]
        """
        if all_paths is None:
            pathname = os.path.join(root, filename_glob)
            all_paths = natsorted(list(glob.glob(pathname, recursive=True)))
        else:
            all_paths = natsorted(all_paths)
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            for keyword in exclude:
                all_paths = [val for val in all_paths if keyword not in val]

        means = []
        stds = []
        if num_bands is None:
            num_bands = rasterio.open(all_paths[0]).count - 1  # excluding QA_pixel
        for i in tqdm(range(num_bands)):
            images = []
            for filepath in all_paths:
                try:
                    img = rasterio.open(filepath).read(i + 1)
                    images.append(img.flatten())
                except Exception:
                    pass

            images = np.concatenate(images, axis=0)
            means.append(images.mean())
            stds.append(images.std())

        return means, stds

    def transform(
        self,
        img: torch.Tensor,
        mask: Union[None, torch.Tensor],
        is_training: Optional[bool] = True,
        p_hflip: Optional[float] = 0.5,
        p_vflip: Optional[float] = 0.5,
        p_mix_patch: Optional[float] = 1.0,
        repeat: Optional[int] = 1,
    ) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        """
        Apply transformations to the input Landsat image and land cover.

        :param img: Input image tensor.
        :type img: torch.Tensor
        :param mask: Land cover tensor.
        :type mask: Optional[torch.Tensor]
        :param is_training: Flag indicating whether the transformation is applied during training.
        :type is_training: Optional[bool]
        :param p_hflip: Probability of horizontal flip.
        :type p_hflip: Optional[float]
        :param p_vflip: Probability of vertical flip.
        :type p_vflip: Optional[float]
        :param p_mix_patch: Probability of mixing patches.
        :type p_mix_patch: Optional[float]
        :param repeat: Number of times to repeat the transformations.
        :type repeat: Optional[int]
        :return: Transformed image and land cover tensors.
        :rtype: Tuple[torch.Tensor, Union[None, torch.Tensor]]
        """
        img = self.transforms(img)

        if is_training:
            for _ in range(repeat):
                if random.random() < p_hflip:
                    img = F.hflip(img)
                    mask = F.hflip(mask)

                if random.random() < p_vflip:
                    img = F.vflip(img)
                    mask = F.vflip(mask)

                if random.random() < p_mix_patch:
                    img, mask = mix_patch(img, mask)

        return img, mask

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        """
        Callable method to apply transformations.

        :param args: Positional arguments.
        :type args: Any
        :param kwargs: Keyword arguments.
        :type kwargs: Any
        :return: Transformed Landsat image and land cover tensors.
        :rtype: Tuple[torch.Tensor, Union[None, torch.Tensor]]
        """
        return self.transform(*args, **kwargs)


class MSSTransformer(LandsatTransformer):
    """
    Input Preprocessor for Landsat data from MSS sensor (Landsat 1 - 5).
    """
    def __init__(self, means=MEAN_MSS, stds=STD_MSS):
        """
        Initialize input preprocessor for Landsat data from MSS sensor (Landsat 1 - 5).

        :param means: List of mean values for normalization.
        :type means: List[float]
        :param stds: List of standard deviation values for normalization.
        :type stds: List[float]
        """
        ndbi_indices = None  # SWIR is unavailable for MSS sensors
        ndvi_indices = {"index_nir": 3, "index_red": 1}
        bui_indices = None
        super().__init__(means, stds, ndbi_indices, ndvi_indices, bui_indices)

class TMTransformer(LandsatTransformer):
    """
    Input Preprocessor for Landsat data from TM sensor (Landsat 4 - 7).
    """
    def __init__(self, means=MEAN_TM, stds=STD_TM):
        """
        Initialize input preprocessor for Landsat data from TM sensor (Landsat 4 - 7).

        :param means: List of mean values for normalization.
        :type means: List[float]
        :param stds: List of standard deviation values for normalization.
        :type stds: List[float]
        """
        ndbi_indices = {"index_swir": 4, "index_nir": 3}
        ndvi_indices = {"index_nir": 3, "index_red": 2}
        bui_indices = {"index_a": -2, "index_b": -1}
        super().__init__(means, stds, ndbi_indices, ndvi_indices, bui_indices)

class OLITIRSTransformer(LandsatTransformer):
    """
    Input Preprocessor for Landsat data from OLI/TIRS sensor (Landsat 8 - 9).
    """
    def __init__(self, means=MEAN_OLITIRS, stds=STD_OLITIRS):
        """
        Initialize input preprocessor for Landsat data from OLI/TIRS sensor (Landsat 8 - 9).

        :param means: List of mean values for normalization.
        :type means: List[float]
        :param stds: List of standard deviation values for normalization.
        :type stds: List[float]
        """
        ndbi_indices = {"index_swir": 5, "index_nir": 4}
        ndvi_indices = {"index_nir": 4, "index_red": 3}
        bui_indices = {"index_a": -2, "index_b": -1}
        super().__init__(means, stds, ndbi_indices, ndvi_indices, bui_indices)