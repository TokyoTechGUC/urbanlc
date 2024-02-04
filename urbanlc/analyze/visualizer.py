import os
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
from .constant import LANDSAT_RGB, ESA2021_LABEL
import ffmpeg

sns.set_theme()

def get_esa_colormap():
    """
    Retrieve the colormap and normalization for ESA (European Space Agency) WorldCover 2021 v200 labels.

    This function creates a colormap and normalization for ESA labels based on the ESA2021_LABEL dictionary.
    The colormap is configured with "black" as the color for values under and over the defined bounds.
    The bounds are derived from the keys of the ESA2021_LABEL dictionary.
    The colormap and normalization are then returned.

    :return: Tuple[matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm]
        Colormap for ESA labels.
        Normalization for the colormap.

    :rtype: Tuple[matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm]
    """
    ESA_color = ListedColormap([v[0] for k, v in ESA2021_LABEL.items()])
    ESA_color.set_under("black")
    ESA_color.set_over("black")
    bounds = list(ESA2021_LABEL.keys()) + [101]
    norm = BoundaryNorm(bounds, ESA_color.N)
    return ESA_color, norm


def plot_class_distribution(
    data: np.ndarray,
    normalize: Optional[bool] = True,
    outfile: Optional[str] = None,
    figsize: Optional[Tuple[int]] = (6, 3),
) -> None:
    """
    Plot the class distribution based on input labels.

    This function takes input data, flattens it, and plots the class distribution using a bar plot.
    If normalization is specified, the distribution is normalized. The plot can be saved to a file if an
    output file path is provided. The function does not return any values.

    :param data: Input data for which the class distribution is to be plotted.
    :type data: numpy.ndarray
    :param normalize: If True, normalize the class distribution such that the sum is 1.0. Defaults to True.
    :type normalize: bool, optional
    :param outfile: File path to save the plot. If None, the plot is not saved. Defaults to None.
    :type outfile: str, optional
    :param figsize: Size of the plot figure. Defaults to (6, 3).
    :type figsize: tuple, optional

    :return: None

    :rtype: None
    """
    data = data.flatten()
    fig, ax = plt.subplots(figsize=figsize)

    count = Counter(data)
    df = pd.DataFrame.from_dict(count, orient="index").sort_index()
    if normalize:
        df[0] = df[0] / df[0].sum()

    df.plot(kind="bar", ax=ax)
    ax.get_legend().remove()
    ax.set_ylim([0, 1])
    ax.set_title("Class distribution")
    fig.autofmt_xdate(rotation=0, ha="center")

    # Annotate each bar with its height
    for p in ax.patches:
        ax.annotate(
            str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.05)
        )

    # Save the plot if an output file path is provided
    if outfile is not None:
        os.makedirs(Path(outfile).parent, exist_ok=True)
        fig.savefig(outfile, dpi=300)


def show_esa_label() -> None:
    """
    Display the ESA (European Space Agency) WorldCover 2021 v200 label colormap and descriptions.

    This function creates a visual representation of the ESA label colormap and their descriptions.
    The colormap and descriptions are retrieved using the get_esa_colormap function.
    The resulting image is displayed with labels and a secondary x-axis for the label values.

    :return: None

    :rtype: None
    """
    labels = np.array([list(ESA2021_LABEL.keys())])
    descriptions = [v[1] for v in ESA2021_LABEL.values()]

    # Get ESA colormap and normalization
    ESA_color, norm = get_esa_colormap()

    # Create a figure and axis for displaying the label colormap
    fig, ax = plt.subplots(figsize=(300, 1))
    ax.imshow(labels, cmap=ESA_color, norm=norm)

    # Set x-axis ticks and labels based on descriptions
    ax.set_xticks(np.arange(len(descriptions)))
    ax.set_xticklabels(descriptions, fontsize=10)

    # Hide grid and y-axis
    ax.grid(False)
    ax.get_yaxis().set_visible(False)

    # Create a secondary x-axis at the top with label values
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(np.arange(len(descriptions)))
    ax2.set_xticklabels(labels[0], fontsize=10)
    ax2.tick_params(top=False)


def plot_land_cover(
    img: np.ndarray,
    ax: plt.Axes,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot land cover data on a given matplotlib axis.

    This function takes land cover data, an axis, and optional parameters to plot the data using a specified colormap.
    The plot can be saved to a file if an output file path is provided. The function does not return any values.

    :param img: Land cover data to be plotted.
    :type img: numpy.ndarray
    :param ax: Matplotlib axis where the land cover plot will be displayed.
    :type ax: matplotlib.axes.Axes
    :param save_path: File path to save the plot. If None, the plot is not saved. Defaults to None.
    :type save_path: str, optional
    :param title: Title for the plot. Defaults to None.
    :type title: str, optional

    :return: None

    :rtype: None
    """
    if img.shape[0] == 1:
        img = img.transpose(1, 2, 0)

    # Get ESA colormap and normalization
    ESA_color, norm = get_esa_colormap()

    # Display land cover data on the given axis
    ax.imshow(img, cmap=ESA_color, norm=norm, interpolation="none")
    
    # Configure axis properties
    ax.grid(False)
    ax.axis("off")
    
    # Set title if provided
    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Save the plot if an output file path is provided
    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_landsat(
    img: np.ndarray,
    dataset: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    ax: plt.Axes = None,
) -> None:
    """
    Plot RGB band of the given Landsat image on a given matplotlib axis.

    This function takes Landsat imagery data, a dataset name for RGB mapping, an axis, and optional parameters to plot
    RGB band of the data using a specified RGB mapping. The plot can be saved to a file if an output file path is provided.
    The function does not return any values.

    :param img: Landsat image data to be plotted.
    :type img: numpy.ndarray
    :param dataset: Name of the Landsat dataset (e.g., 'landsat8') for RGB mapping.
    :type dataset: str
    :param save_path: File path to save the plot. If None, the plot is not saved. Defaults to None.
    :type save_path: str, optional
    :param title: Title for the plot. Defaults to None.
    :type title: str, optional
    :param ax: Matplotlib axis where the Landsat plot will be displayed. Defaults to None.
    :type ax: matplotlib.axes.Axes, optional

    :return: None

    :rtype: None
    """
    assert dataset in LANDSAT_RGB
    assert ax is not None
    if np.argmin(img.shape) == 0:
        img = img.transpose(1, 2, 0)

    # Display Landsat imagery data on the given axis using RGB mapping
    ax.imshow(img[:, :, LANDSAT_RGB[dataset]])
    
    # Configure axis properties
    ax.grid(False)
    ax.axis("off")
    
    # Set title if provided
    if title is not None:
        assert isinstance(title, str)
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Save the plot if an output file path is provided
    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_change(
    img_paths: Optional[List[str]] = None,
    root: Optional[str] = None,
    framerate: Optional[float] = 1,
    save_path: Optional[str] = "output.mp4",
) -> None:
    """
    Create a video plot to visualize changes over time based on image paths or a root directory.

    This function creates a video plot to visualize changes over time either based on a list of image paths or
    images within a root directory. The framerate of the output video can be specified, and the resulting video is saved
    to the specified file path.
    
    :param img_paths: List of image paths for creating the video. Defaults to None.
    :type img_paths: List[str], optional
    :param root: Root directory containing images to create the video. Defaults to None.
    :type root: str, optional
    :param framerate: Framerate of the output video. Defaults to 1.
    :type framerate: float, optional
    :param save_path: File path to save the output video. Defaults to "output.mp4".
    :type save_path: str, optional

    :return: None

    :rtype: None
    """
    assert not ((img_paths is None) and (root is None))
    
    # Remove the existing file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    # Create video based on image paths in a root directory
    if root is not None:
        input = os.path.join(root, "*.png")
        (
            ffmpeg.input(input, pattern_type="glob", framerate=framerate)
            .output(save_path)
            .run()
        )
    # Create video based on a list of image paths
    else:
        # Write image paths to a temporary text file
        content = [f"file {img_path}'\n" for img_path in img_paths]
        with open("temp.txt", "w") as f:
            f.writelines(content)

        # Concatenate images using the temporary text file
        (
            ffmpeg.input("temp.txt", r=str(framerate), f="concat", safe="0")
            .output(save_path)
            .run()
        )

        # Remove the temporary text file
        os.remove("temp.txt")


def visualize_data_batch(
    images: torch.Tensor, gts: torch.Tensor, dataset: str, ax: plt.Axes
) -> None:
    """
    Visualize a batch of data containing Landsat images and corresponding ground truth land cover maps.

    This function takes a batch of Landsat images, a batch of ground truth land cover maps,
    the name of the Landsat dataset for RGB mapping, and a matplotlib axis.
    It visualizes the Landsat images and ground truth land cover maps side by side on the provided axis.

    :param images: Batch of Landsat images.
    :type images: torch.Tensor
    :param gts: Batch of ground truth land cover maps.
    :type gts: torch.Tensor
    :param dataset: Name of the Landsat dataset (e.g., 'landsat8') for RGB mapping.
    :type dataset: str
    :param ax: Matplotlib axis to display the visualizations.
    :type ax: matplotlib.axes.Axes

    :return: None

    :rtype: None
    """
    for i, (image, gt) in enumerate(zip(images, gts)):
        # Plot Landsat image on even indices
        plot_landsat(image.numpy(), dataset=dataset, ax=ax[2 * i])

        # Plot ground truth land cover map on odd indices
        plot_land_cover(gt.numpy(), ax=ax[2 * i + 1])
