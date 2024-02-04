import rasterio
import numpy as np
from natsort import natsorted
from tqdm.auto import tqdm
from typing import List, Optional, Union, Dict, Any, Tuple

import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
from torchmetrics import F1Score, Accuracy, JaccardIndex

from .base import LCC
from .train_utils import (
    save_checkpoint,
    set_seed,
    load_checkpoint,
    segment_satelite_image,
    combine_prediction
)
from .pipeline_transforms import MSSTransformer, TMTransformer, OLITIRSTransformer
from .dataloader import get_dataloader, parse_paths

try:
    HAVE_WANDB = True
    import wandb
except Exception:
    HAVE_WANDB = False
    warnings.warn("wandb is not installed, deep-learning model training is disabled.")

class DeepLearningLCC(LCC):
    """
        Deep-learning-based Land Cover Classification (LCC) model.
    """

    def __init__(
        self,
        architecture: str,
        model_params: Dict[str, Any],
        device: Optional[str] = None,
        seed: Optional[int] = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Deep Learning LCC model.

        :param architecture: Name of the deep learning model architecture.
        :type architecture: str
        :param model_params: Dictionary containing model-specific parameters.
        :type model_params: Dict[str, Any]
        :param device: Device to use for training (e.g., "cuda" or "cpu").
        :type device: Optional[str]
        :param seed: Seed for reproducibility.
        :type seed: Optional[int]
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """
        set_seed(seed)
        super().__init__(*args, **kwargs)
        self.set_device(device)
        self.build_model(architecture, model_params)
        self.cache = {}

    def set_device(self, device: Union[None, str]) -> None:
        """
        Set the device for training or inference. Defaults to "cuda" if available.

        :param device: Device to use for training (e.g., "cuda" or "cpu").
        :type device: Union[None, str]
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        # print(f"Using {self.device}")

    def build_model(self, architecture: str, model_params: Dict[str, Any]) -> None:
        """
        Build the deep learning model and moves it to the selected device.

        :param architecture: Name of the deep learning model architecture.
        :type architecture: str
        :param model_params: Dictionary containing model-specific parameters.
        :type model_params: Dict[str, Any]
        """
        self.model = getattr(smp, architecture)(**model_params)
        self.model = self.model.to(self.device)
        self.elapsed_epoch = 0

    def to(self, device):
        """
        Move the model to a different device.

        :param device: Device to move the model to.
        :type device: Any
        """

        self.set_device(device)
        self.model = self.model.to(self.device)

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a pre-trained model checkpoint.

        :param checkpoint_path: Path to the pre-trained model.
        :type checkpoint_path: str
        """
        params = {
            "checkpoint_path": checkpoint_path,
            "model": self.model,
            "optimizer": getattr(self, "optimizer", None),
            "scheduler": getattr(self, "scheduler", None),
            "device": self.device,
        }
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.elapsed_epoch,
        ) = load_checkpoint(**params)

    def save_model(self, filename: str, current_epoch: int) -> None:
        """
        Save the current model checkpoint.

        :param filename: Name of the checkpoint file.
        :type filename: str
        :param current_epoch: Current training epoch.
        :type current_epoch: int
        """
        params = {
            "save_dir": self.save_path,
            "name": filename,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "epoch": current_epoch,
        }
        save_checkpoint(**params)

    def setup_trainer(
        self,
        loss_fn_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        scheduler_params: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Set up the trainer with loss functions, optimizer, and scheduler.

        :param loss_fn_params: Dictionary containing loss function parameters.
        :type loss_fn_params: Dict[str, Any]
        :param optimizer_params: Dictionary containing optimizer parameters.
        :type optimizer_params: Dict[str, Any]
        :param scheduler_params: Dictionary containing scheduler parameters.
        :type scheduler_params: Dict[str, Any]
        """
        self.loss_funcs = []
        for name, params in loss_fn_params.items():
            if name == "weights":
                self.loss_weights = params
            else:
                loss_fn = getattr(losses, name)(**params)
                self.loss_funcs.append(loss_fn)

        assert len(self.loss_weights) == len(self.loss_funcs)

        self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_params)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)

    def normalize_class(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Normalize ground-truth labels.

        :param gt: Ground-truth tensor.
        :type gt: torch.Tensor
        :return: Normalized ground-truth tensor.
        :rtype: torch.Tensor
        """
        for i, val in enumerate(self.legends):
            gt[gt == val] = i

        return gt

    def denormalize_class(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Denormalize ground-truth labels.

        :param gt: Normalized ground-truth tensor.
        :type gt: torch.Tensor
        :return: Denormalized ground-truth tensor.
        :rtype: torch.Tensor
        """
        for i, val in enumerate(self.legends[::-1]):
            gt[gt == len(self.legends) - i - 1] = val

        return gt

    def get_metrics(self, mode: str) -> Dict[str, Any]:
        """
        Get metrics for the specified mode (Train or Val).

        :param mode: Training mode ("Train") or validation mode ("Val").
        :type mode: str
        :return: Dictionary of metrics.
        :rtype: Dict[str, Any]
        """
        assert mode in ["Train", "Val"]

        kwargs = {"task": "multiclass", "num_classes": len(self.legends)}
        metrics = {
            f"{mode}/accuracy_micro": Accuracy(**kwargs, average="micro").to(self.device),
            f"{mode}/accuracy_macro": Accuracy(**kwargs, average="macro").to(self.device),
            f"{mode}/f1_macro": F1Score(**kwargs, average="macro").to(self.device),
            f"{mode}/f1_weighted": F1Score(**kwargs, average="weighted").to(self.device),
            f"{mode}/iou": JaccardIndex(**kwargs).to(self.device),
        }
        return metrics

    def train_one_epoch(
        self,
        step: int,
        train_loader: DataLoader,
        epoch: int,
        MAX_EPOCH: int,
        metrics: Dict[str, Any],
        GRADIENT_ACCUMULATION_FACTOR: Optional[int],
    ) -> Tuple[int, Dict[str, Any], float, int]:
        """
        Train the model for one epoch.

        :param step: Current training step.
        :type step: int
        :param train_loader: DataLoader for training.
        :type train_loader: DataLoader
        :param epoch: Current training epoch.
        :type epoch: int
        :param MAX_EPOCH: Maximum number of training epochs.
        :type MAX_EPOCH: int
        :param metrics: Dictionary of metrics.
        :type metrics: Dict[str, Any]
        :param GRADIENT_ACCUMULATION_FACTOR: Gradient accumulation factor.
        :type GRADIENT_ACCUMULATION_FACTOR: Optional[int]
        :return: Updated step, metrics, training loss, and sample count.
        :rtype: Tuple[int, Dict[str, Any], float, int]
        """
        # train for one epoch
        sample_count = 0
        train_loss = 0
        for idx, batch in tqdm(
            enumerate(train_loader),
            desc=f"Epoch: {epoch + 1}/{MAX_EPOCH}. Loop: Train",
            total=len(train_loader),
            position=0,
            leave=True,
        ):
            self.optimizer.zero_grad()

            # get image and ground-truth
            img = batch["image"][:, :-1, :, :].float().to(self.device)
            gt = self.normalize_class(batch["mask"]).to(self.device)

            # data augmentation and esa label normalization
            params = {
                "img": img,
                "mask": gt,
                "is_training": True,
                "p_hflip": 0.5,
                "p_vflip": 0.5,
                "p_mix_patch": 1.0 if epoch < 0.8 * MAX_EPOCH else 0.0,
            }
            img, gt = self.transform_pipeline(**params)
            gt = gt.squeeze()

            # prediction, loss calculation, and average performance tracking
            preds = self.model(img)

            loss = None
            for weight, loss_fn in zip(self.loss_weights, self.loss_funcs):
                if loss is None:
                    loss = weight * loss_fn(preds, gt)
                else:
                    loss = loss + weight * loss_fn(preds, gt)

            if self.use_wandb:
                wandb.log({f"Train/loss": loss.item()}, step=step)
            else:
                print(f"step: {step} loss: {loss.item()}")

            sample_count += img.size(0)
            batch_loss = loss.item()
            train_loss += batch_loss * img.size(0)

            preds = torch.argmax(preds, dim=1)
            for metric in metrics.values():
                _ = metric(preds, gt)

            # weight update (+ gradient accumulation)
            if GRADIENT_ACCUMULATION_FACTOR is None:
                loss.backward()
                self.optimizer.step()
            else:
                loss = loss / GRADIENT_ACCUMULATION_FACTOR
                loss.backward()
                if ((idx + 1) % GRADIENT_ACCUMULATION_FACTOR == 0) or (idx + 1 == len(train_loader)):
                    self.optimizer.step()

            step += 1

        return step, metrics, train_loss, sample_count

    def train(
        self,
        dataloader_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
        validate_params: Optional[Dict[str, Any]],
        logger_params: Optional[Dict[str, Any]],
    ) -> None:
        """
        Train the deep learning model.

        :param dataloader_params: Parameters for data loading.
        :type dataloader_params: Dict[str, Any]
        :param trainer_params: Parameters for the trainer.
        :type trainer_params: Dict[str, Any]
        :param validate_params: Parameters for validation.
        :type validate_params: Dict[str, Any]
        :param logger_params: Parameters for the logger.
        :type logger_params: Dict[str, Any]
        """
        # initialize logger
        self.use_wandb = HAVE_WANDB and ("PROJECT" in logger_params)
        if self.use_wandb:
            config_list = [dataloader_params, trainer_params, validate_params]
            config = {k: v for d in config_list for k, v in d.items()}
            wandb.init(
                project=logger_params["PROJECT"],
                name=logger_params["NAME"],
                config=config,
                settings=wandb.Settings(start_method="fork"),
            )

        # loss func, optimizer, scheduler
        self.setup_trainer(**trainer_params)

        # training scores
        metrics = self.get_metrics(mode="Train")
        main_metrics = "Val/" + "f1_macro"
        keys = [metric.replace("Train", "Val") for metric in metrics.keys()]
        best_scores = dict.fromkeys(keys, -1)

        MAX_EPOCH = trainer_params["epoch"]
        GRADIENT_ACCUMULATION_FACTOR = trainer_params["gradient_accumulation_factor"]
        step = 0
        loader_remaining = 0

        for epoch in range(MAX_EPOCH):
            # prepare data
            if loader_remaining == 0:
                train_loaders = get_dataloader(**dataloader_params)
                loader_remaining = len(train_loaders)
            
            train_loader = train_loaders[loader_remaining - 1]

            # train for one epoch
            if self.use_wandb:
                wandb.log({"Epoch": epoch}, step=step)
            else:
                print(f"Epoch {epoch}")

            step, metrics, train_loss, sample_count = \
                    self.train_one_epoch(step, train_loader, epoch, MAX_EPOCH, metrics, GRADIENT_ACCUMULATION_FACTOR)

            self.scheduler.step()

            # validate model and log result
            train_results = {}
            for metric_name, metric in metrics.items():
                score = metric.compute()
                train_results[metric_name] = score.detach().cpu().numpy()
                metric.reset()

            if validate_params is not None:
                val_results = self.validate(**validate_params)

                if (
                    best_scores[main_metrics] == -1
                    or val_results[main_metrics] >= best_scores[main_metrics]
                ):
                    best_scores = val_results.copy()
                    self.save_model(f"{logger_params['NAME']}_best", epoch)

            if self.use_wandb:
                result = train_results.copy()
                result.update(val_results)
                result["Train/average_loss"] = train_loss / sample_count
                wandb.log(result, step=step)
            
            loader_remaining -= 1

            self.save_model(f"{logger_params['NAME']}_epoch{epoch}", epoch)
            self.save_model(f"{logger_params['NAME']}_latest", epoch)

        if self.use_wandb:
            wandb.finish()

    def validate(self, root: str = None, img_glob: str = None, gt_glob: str = None,
        img_paths: Optional[List[str]] = None, gt_paths: Optional[List[str]] = None, cache: bool = True) -> Dict[str, Any]:
        """
        Validate the deep learning model.

        :param root: Root directory for data.
        :type root: str, optional
        :param img_glob: Image file pattern.
        :type img_glob: str, optional
        :param gt_glob: Ground truth file pattern.
        :type gt_glob: str, optional
        :param img_paths: List of image paths.
        :type img_paths: List[str], optional
        :param gt_paths: List of ground truth paths.
        :type gt_paths: List[str], optional
        :param cache: Whether to use caching.
        :type cache: bool, optional
        :return: Dictionary of validation results.
        :rtype: Dict[str, Any]
        """
        # get all paths
        img_paths = parse_paths(root, img_glob) if img_paths is None else natsorted(img_paths)
        gt_paths = parse_paths(root, gt_glob) if gt_paths is None else natsorted(gt_paths)
        assert len(img_paths) == len(gt_paths)
        
        # validation scores
        val_metrics = self.get_metrics(mode="Val")

        # validate
        self.model.eval()
        val_loss = 0
        for img_path, gt_path in tqdm(zip(img_paths, gt_paths), total=len(img_paths)):
            preds, preds_prob = self.infer(img_path, convert_numpy=False, cache=cache, return_prob=True)
            preds = preds.to(self.device)
            preds_prob = preds_prob.to(self.device)

            if cache:
                gt = self.cache[gt_path] if gt_path in self.cache else rasterio.open(gt_path).read()
                self.cache[gt_path] = gt
            else:
                gt = rasterio.open(gt_path).read()
            gt = torch.from_numpy(gt)
            gt = self.normalize_class(gt).to(self.device)

            val_loss += self.loss_funcs[0](preds_prob, gt).item()
            for metric_name, metric in val_metrics.items():
                _ = metric(preds, gt)

        val_results = {"Val/average_loss": val_loss / len(img_paths)}
        for metric_name, metric in val_metrics.items():
            score = metric.compute()
            val_results[metric_name] = score.detach().cpu().numpy()

        self.model.train()
        return val_results

    def infer(
        self, img_path: str, convert_numpy: Optional[bool] = True, cache=False, return_prob=False, stride=None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform inference using the deep learning model.

        :param img_path: Path to the input image.
        :type img_path: str
        :param convert_numpy: Whether to convert the result to NumPy array.
        :type convert_numpy: bool, optional
        :param cache: Whether to use caching.
        :type cache: bool, optional
        :param return_prob: Whether to return probability map.
        :type return_prob: bool, optional
        :param stride: Stride for inference.
        :type stride: int, optional
        :return: Inference result.
        :rtype: Union[np.ndarray, torch.Tensor]
        """
        if cache:
            img = self.cache[img_path] if img_path in self.cache else rasterio.open(img_path).read()
            self.cache[img_path] = img
        else:
            img = rasterio.open(img_path).read()

        input_img = torch.from_numpy(img)
        input_patches, bounding_boxes = segment_satelite_image(input_img, stride=stride)
        input_patches = torch.cat(input_patches, axis=0)
        input_patches = input_patches[:, :-1, :, :]

        input_patches, _ = self.transform_pipeline(
            input_patches, None, is_training=False
        )
        input_patches = input_patches.float()

        # inference
        self.model.eval()
        with torch.no_grad():
            preds = [
                self.model(patch.unsqueeze(0).to(self.device)).cpu()
                for patch in input_patches
            ]

        preds_patches = torch.stack(preds, axis=1)
        preds_prob = combine_prediction(preds_patches, bounding_boxes, img.shape[1:])
        preds = torch.argmax(preds_prob, axis=1)
        preds = preds.cpu().numpy() if convert_numpy else preds

        if return_prob:
            return preds, preds_prob
        else:
            return preds


class MSSDeepLearning(DeepLearningLCC):
    """
    Deep-learning-based LCC for Landsat 1 - 5 (using MSS sensor).
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize deep-learning-based LCC for Landsat 1 - 5 (using MSS sensor).

        :param args: Additional positional arguments.
        :type args: Tuple
        :param kwargs: Additional keyword arguments.
        :type kwargs: Dict
        """
        self.transform_pipeline = MSSTransformer()
        super().__init__(*args, **kwargs)


class TMDeepLearning(DeepLearningLCC):
    """
    Deep-learning-based LCC for Landsat 4 - 7 (using TM sensor).
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize deep-learning-based LCC for Landsat 4 - 7 (using TM sensor).

        :param args: Additional positional arguments.
        :type args: Tuple
        :param kwargs: Additional keyword arguments.
        :type kwargs: Dict
        """
        self.transform_pipeline = TMTransformer()
        super().__init__(*args, **kwargs)


class OLITIRSDeepLearning(DeepLearningLCC):
    """
    Deep-learning-based LCC for Landsat 8 - 9 (using OLI/TIRS sensor).
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize deep-learning-based LCC for Landsat 8 - 9 (using OLI/TIRS sensor).

        :param args: Additional positional arguments.
        :type args: Tuple
        :param kwargs: Additional keyword arguments.
        :type kwargs: Dict
        """
        self.transform_pipeline = OLITIRSTransformer()
        super().__init__(*args, **kwargs)
