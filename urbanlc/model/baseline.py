import os
import pickle
from pathlib import Path

from tqdm.auto import tqdm
from itertools import zip_longest

import numpy as np
import rasterio

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report

import xgboost
from xgboost import XGBClassifier
from typing import List, Optional, Tuple, Dict, Any, Union

from .pipeline_transforms import compute_NDBI, compute_NDVI, compute_BUI

from .base import LCC


class BaselineLCC(LCC):
    """
    BaselineLCC abstract class for LCC with classical ML
    """
    def __init__(self, model_name: str, model_params: dict, *args, **kwargs) -> None:
        """
        Initialize the BaselineLCC model.

        :param model_name: Name of the model ("xgb", "svm", or "logistic_regression").
        :type model_name: str
        :param model_params: Parameters for initializing the specific model.
        :type model_params: dict
        :param args: Additional arguments passed to the superclass.
        :param kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.build_model(model_params)

    def build_model(self, model_params: dict) -> None:
        """
        Build the specific model based on the provided model name and parameters.

        :param model_params: Parameters for initializing the specific model.
        :type model_params: dict
        """
        if self.model_name == "xgb":
            self.model = XGBClassifier(**model_params)
        elif self.model_name == "svm":
            self.model = make_pipeline(StandardScaler(), SVC(**model_params))
        elif self.model_name == "logistic_regression":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**model_params))

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a pre-trained model from a checkpoint file.

        :param checkpoint_path: Path to the checkpoint file.
        :type checkpoint_path: str
        :rtype: None
        """
        try:
            with open(checkpoint_path, "rb") as f:
                temp, temp2 = pickle.load(f)

            if not isinstance(temp, xgboost.sklearn.XGBClassifier):
                assert isinstance(temp, dict) and ("estimator" in temp)
                temp = temp["estimator"][np.argmax(temp["test_score"])]

            self.model = temp
            self.legends = temp2
            self.construct_transform_map()
            # logger.info(f"Successfully loaded model at {checkpoint_path}")
            print(f"Successfully loaded model at {checkpoint_path}")
        except Exception as e:
            raise (e)

    def save_model(self, data: Tuple[Any, List[int]]) -> None:
        """
        Save the current model and related information to a checkpoint file.

        :param data: Tuple containing the model-related information to be saved.
        :type data: Tuple[Any, List[int]]
        :rtype: None
        """
        try:
            os.makedirs(Path(self.save_path).parent, exist_ok=True)
            with open(self.save_path, "wb") as f:
                pickle.dump(data, f)
            # logger.info(f"Successfully saved model at {self.save_path}")
            print(f"Successfully saved model at {self.save_path}")
        except Exception as e:
            raise (e)

    # TODO: can we optimize this?
    def update_transform_map(self, y_train: np.ndarray) -> None:
        """
        Update the transform map based on the provided training labels.

        :param y_train: Training labels.
        :type y_train: np.ndarray
        :rtype: None
        """
        seen_labels = np.unique(y_train)
        unseen_labels = np.array([id for id in self.legends if id not in seen_labels])
        self.legends = np.concatenate([seen_labels, unseen_labels])
        self.construct_transform_map()

    def transform_pipeline(self, img: np.ndarray) -> np.ndarray:
        """
        Transform the input image using specified transformations.

        :param img: Input image.
        :type img: np.ndarray

        :return: Transformed image.
        :rtype: np.ndarray
        """
        all_bands = [img]
        if self.ndbi_indices is not None:
            all_bands.append(compute_NDBI(img, **self.ndbi_indices))
        if self.ndvi_indices is not None:
            all_bands.append(compute_NDVI(img, **self.ndvi_indices))
        if self.bui_indices is not None:
            all_bands.append(compute_BUI(img, **self.bui_indices))

        img = np.concatenate(all_bands, axis=0)
        img = img.reshape(img.shape[0], -1)
        return img

    def retrieve_images(
        self,
        img_paths: List[str],
        gt_paths: List[Any] = [],
        return_size: Optional[bool] = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, List[Tuple[int]]], Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve and preprocess Landsat images and ground truth land cover data from specified paths for training/inference.

        :param img_paths: List of paths to Landsat image files.
        :type img_paths: List[str]
        :param gt_paths: List of paths to ground truth land cover files.
        :type gt_paths: List[Any]
        :param return_size: Whether to return the original sizes of Landsat images.
        :type return_size: Optional[bool]

        :return: Tuple containing retrieved images and ground truth land cover data.
        :rtype: Union[Tuple[np.ndarray, np.ndarray, List[Tuple[int]]], Tuple[np.ndarray, np.ndarray]]
        """
        print("Retrieving images...")
        img_paths = [img_paths] if isinstance(img_paths, str) else img_paths
        gt_paths = [gt_paths] if isinstance(gt_paths, str) else gt_paths

        images = []
        gts = []
        original_size = []
        input_paths = list(zip_longest(img_paths, gt_paths, fillvalue=None))
        for img_path, gt_path in tqdm(input_paths, desc="Preparing data"):
            assert os.path.exists(img_path), f"{img_path} does not exist"

            img = rasterio.open(img_path).read()[:-1]  # filter out QA mask
            original_size.append((1, img.shape[1], img.shape[2]))

            img = self.transform_pipeline(img)
            images.append(img)

            if gt_path is not None:
                assert os.path.exists(gt_path), f"{gt_path} does not exist"

                gt = rasterio.open(gt_path).read()
                gt = gt.reshape(gt.shape[0], -1)
                gts.append(gt)

        if return_size:
            return images, gts, original_size
        else:
            return images, gts

    def train(
        self,
        img_paths: List[str],
        gt_paths: List[str],
        enable_cv: Optional[bool] = True,
        cross_validate_params: Optional[Dict[str, Any]] = None,
        train_size: Optional[float] = 1.0,
    ) -> None:
        """
        Train the model using specified Landsat image and ground truth land cover paths, with optional cross-validation.

        :param img_paths: List of paths to Landsat image files.
        :type img_paths: List[str]
        :param gt_paths: List of paths to ground truth land cover files.
        :type gt_paths: List[str]
        :param enable_cv: Whether to perform cross-validation.
        :type enable_cv: Optional[bool]
        :param cross_validate_params: Parameters for cross-validation.
        :type cross_validate_params: Optional[Dict[str, Any]]
        :param train_size: Ratio of observation points (pixels) to be used as the training data.
        :type train_size: Optional[float]
        :rtype: None
        """
        images, gts = self.retrieve_images(img_paths, gt_paths)

        images = np.concatenate(images, axis=1)
        gts = np.concatenate(gts, axis=1)

        X_train = images.transpose()
        y_train = gts.squeeze()

        if train_size > 1.0:
            train_ratio = float(train_size) / len(X_train)
            # print(train_ratio)
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=train_ratio,
                random_state=0,
                stratify=y_train,
            )
        elif train_size < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=train_size,
                random_state=0,
                stratify=y_train,
            )

        self.update_transform_map(y_train)
        y_train = np.vectorize(self.transform_func)(y_train)
        print(f"#samples = {X_train.shape}")

        if enable_cv:
            # logger.info("Training XGBoost using cross-validation...")
            print(f"Training {self.model_name} using cross-validation...")
            cv_results = cross_validate(
                self.model, X=X_train, y=y_train, **cross_validate_params
            )

            self.model = cv_results["estimator"][np.argmax(cv_results["test_score"])]
            # logger.info(f"validation score: {cv_results['test_score']}")
            print(f"Validation score: {cv_results['test_score']}")
            self.save_model((cv_results, self.legends))
        else:
            # logger.info("Training XGBoost...")
            print(f"Training {self.model_name}...")
            self.model.fit(X_train, y_train)
            self.save_model((self.model, self.legends))

    def validate(
        self, img_paths: List[str], gt_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Validate the model using specified Landsat image and ground truth land cover paths.

        :param img_paths: List of paths to Landsat image files.
        :type img_paths: List[str]
        :param gt_paths: List of paths to ground truth land cover files.
        :type gt_paths: List[str]

        :return: List of dictionaries containing classification report results.
        :rtype: List[Dict[str, Any]]
        """
        images, gts = self.retrieve_images(img_paths, gt_paths)

        results = []
        for img, gt in tqdm(zip(images, gts), total=len(images), desc="Validating"):
            X = img.transpose()
            y = gt.squeeze()
            y = np.vectorize(self.transform_func)(y)

            preds = self.model.predict(X)
            result = classification_report(
                y,
                preds,
                labels=list(range(len(self.legends))),
                target_names=self.class_names,
                output_dict=True,
            )
            results.append(result)

        return results

    def infer(self, img_paths: List[str]) -> List[np.ndarray]:
        """
        Perform inference on specified Landsat image paths and returns the inferred land cover maps.

        :param img_paths: List of paths to Landsat image files.
        :type img_paths: List[str]

        :return: List of inferred land cover maps.
        :rtype: List[np.ndarray]
        """
        images, _, original_size = self.retrieve_images(img_paths, return_size=True)
        preds = []
        for img, size in tqdm(zip(images, original_size), total=len(original_size)):
            X = img.transpose()
            pred = self.model.predict(X)
            pred = np.vectorize(self.inv_transform_func)(pred)
            pred = pred.reshape(original_size[0])
            preds.append(pred)

        return preds



class MSSBaseline(BaselineLCC):
    """
    MSSBaseline class for LCC with classical ML using data from Landsat 1 - 5 (MSS sensor)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize MSSBaseline model, inheriting from the BaselineLCC model.

        :param args: Additional arguments passed to the superclass.
        :param kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.ndbi_indices = None  # SWIR is unavailable for MSS sensors
        self.ndvi_indices = {"index_nir": 3, "index_red": 1}
        self.bui_indices = None


class TMBaseline(BaselineLCC):
    """
    TMBaseline class for LCC with classical ML using data from Landsat 4 - 7 (TM sensor)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize TMBaseline model, inheriting from the BaselineLCC model.

        :param args: Additional arguments passed to the superclass.
        :param kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.ndbi_indices = {"index_swir": 4, "index_nir": 3}
        self.ndvi_indices = {"index_nir": 3, "index_red": 2}
        self.bui_indices = {"index_a": -2, "index_b": -1}


class OLI_TIRSBaseline(BaselineLCC):
    """
    OLI_TIRSBaseline class for LCC with classical ML using data from Landsat 8 - 9 (OLI/TIRS sensor)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize OLI_TIRSBaseline model, inheriting from the BaselineLCC model.

        :param args: Additional arguments passed to the superclass.
        :param kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.ndbi_indices = {"index_swir": 5, "index_nir": 4}
        self.ndvi_indices = {"index_nir": 4, "index_red": 3}
        self.bui_indices = {"index_a": -2, "index_b": -1}
