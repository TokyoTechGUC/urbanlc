# ref: http://gsp.humboldt.edu/olm/Courses/GSP_216/lessons/accuracy/metrics.html
import numpy as np
import os
import rasterio
import torch
from torchmetrics import ConfusionMatrix
from typing import List, Optional

from ..utils import open_at_size, open_at_scale
from .constant import ESA1992_map, ESA2021_map, ESA2021_CLASSES

import numpy as np
import os
import rasterio
import torch
from torchmetrics import ConfusionMatrix

from ..utils import open_at_size, open_at_scale
from .constant import ESA1992_map, ESA2021_map, ESA2021_CLASSES

def confusion_matrix(
    pred_path: str,
    gt_path: str,
    mapper_gt: List = ESA1992_map,
    mapper_pred: List = ESA2021_map,
    gt_downscale_factor: Optional[float] = None,
    use_pred_as_ref: Optional[bool] = False,
) -> np.ndarray:
    """
    Calculate the confusion matrix based on predicted and ground truth data.

    This function takes paths to predicted and ground truth data, optionally downscales the ground truth data,
    and then calculates the confusion matrix. The super-class mappings are applied, and the result is returned as a numpy array.

    :param pred_path: Path to the predicted data.
    :type pred_path: str
    :param gt_path: Path to the ground truth data.
    :type gt_path: str
    :param mapper_gt: Super-class mapping dictionary for ground truth classes. Defaults to ESA1992_map.
    :type mapper_gt: List, optional
    :param mapper_pred: Super-class mapping dictionary for predicted classes. Defaults to ESA2021_map.
    :type mapper_pred: List, optional
    :param gt_downscale_factor: Downscale factor for ground truth data. Defaults to None.
    :type gt_downscale_factor: float, optional
    :param use_pred_as_ref: If True, use predicted data as reference; otherwise, use ground truth. Defaults to False.
    :type use_pred_as_ref: bool, optional

    :return: Confusion matrix.
    :rtype: np.ndarray
    """

    assert os.path.exists(pred_path)
    if not os.path.exists(gt_path):
        return None
    else:
        if not use_pred_as_ref:
            if gt_downscale_factor is not None:
                gt = open_at_scale(gt_path, gt_downscale_factor)
            else:
                gt = rasterio.open(gt_path).read()
            pred = open_at_size(pred_path, gt)
        else:
            if gt_downscale_factor is not None:
                pred = open_at_scale(pred_path, gt_downscale_factor)
            else:
                pred = rasterio.open(pred_path).read()
            gt = open_at_size(gt_path, pred)

        assert gt.shape == pred.shape

        # Apply class mappings
        gt = np.vectorize(lambda x: mapper_gt[x])(gt)
        pred = np.vectorize(lambda x: mapper_pred[x])(pred)

        gt = torch.from_numpy(gt)
        pred = torch.from_numpy(pred)
        assert gt.shape == pred.shape

        # Create ConfusionMatrix object
        CONFUSION_MATRIX = ConfusionMatrix(
            task="multiclass",
            num_classes=len(set(list(mapper_pred.values()))),
            ignore_index=-1,
        )

        # Calculate and return the confusion matrix
        return CONFUSION_MATRIX(pred, gt).numpy().transpose()


def accuracy(m: np.ndarray) -> float:
    """
    Calculate accuracy from a confusion matrix.

    :param m: Confusion matrix.
    :type m: np.ndarray

    :return: Accuracy calculated as the sum of diagonal elements divided by the sum of all elements in the matrix.
    :rtype: float
    """

    return m.diagonal().sum() / m.sum()


def producer_accuracy(m: np.ndarray) -> np.ndarray:
    """
    Calculate producer's accuracy from a confusion matrix.

    :param m: Confusion matrix.
    :type m: np.ndarray

    :return: Producer's accuracy calculated as the diagonal elements divided by the sum of each column in the matrix.
    :rtype: np.ndarray
    """

    return m.diagonal() / m.sum(axis=0)


def user_accuracy(m: np.ndarray) -> np.ndarray:
    """
    Calculate user's accuracy from a confusion matrix.

    :param m: Confusion matrix.
    :type m: np.ndarray

    :return: User's accuracy calculated as the diagonal elements divided by the sum of each row in the matrix.
    :rtype: np.ndarray
    """

    return m.diagonal() / m.sum(axis=1)


def cohen_kappa(m: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa coefficient from a confusion matrix.

    :param m: Confusion matrix.
    :type m: np.ndarray

    :return: Cohen's Kappa coefficient, a measure of agreement between observers, adjusted for chance.
    :rtype: float

    The function computes row and column totals, as well as the observed and expected probabilities.
    Finally, Cohen's Kappa coefficient is calculated and returned.
    """

    n = m.sum()

    # Calculate row and column totals
    row_totals = m.sum(axis=1)
    col_totals = m.sum(axis=0)

    # Calculate observed (p0) and expected (pe) probabilities
    p0 = np.trace(m) / n
    pe = row_totals * col_totals / (n ** 2)

    # Calculate Cohen's Kappa coefficient
    kappa = (p0 - pe.sum()) / (1 - pe.sum())
    return kappa


def get_class_distribution(
    path: str,
    downsample_scale: float,
    indices: List = ESA2021_CLASSES,
) -> List:
    """
    Calculate the class distribution of land cover map in a specified data path at a specified resolution.

    :param path: Path to the data.
    :type path: str
    :param downsample_scale: Downscaling ratio for the land cover map.
    :type downsample_scale: float
    :param indices: List of indices representing classes. Defaults to ESA2021_CLASSES.
    :type indices: List, optional

    :return: Class distribution as a list of proportions.
    :rtype: List
    """

    data = open_at_scale(path, downsample_scale=downsample_scale).flatten()

    # Calculate class distribution
    dist = [len(data[data == index]) / len(data) for index in indices]
    return dist


# m = [[21, 6, 0], [5, 31, 1], [7, 2, 22]]
# m = np.array(m)
# print(accuracy(m))
# print(producer_accuracy(m))
# print(user_accuracy(m))