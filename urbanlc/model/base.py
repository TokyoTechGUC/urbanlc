from abc import ABC, abstractmethod
from typing import List, Optional

ESA_LEGENDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
ESA_CLASS_NAME = [
    "Tree",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse vegetation2",
    "Snow and ice",
    "Permanent water bodies",
    "Herbaceous wetland",
    "Mangroves",
    "Moss and lichen",
]


class LCC(ABC):
    """
        Abstract base class for Land Cover Classification (LCC) models.
    """

    def __init__(
        self,
        save_path: str,
        legends: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        This constructor initializes the basic attributes of the LCC model,
        including the save path, legends, and class names. If legends and class_names are not provided,
        it uses default values from ESA2021 (11 classes).

        :param save_path: File path to save the model.
        :param legends: List of class labels or legends (optional).
        :param class_names: List of class names corresponding to legends (optional).
        """
        self.save_path = save_path

        if (legends is None) and (class_names is None):
            # logger.info("Use legends from ESA2021 (11 classes) as default")
            print("Use legends from ESA2021 (11 classes) as default")
            self.legends = ESA_LEGENDS
            self.class_names = ESA_CLASS_NAME
        else:
            self.legends = legends
            self.class_names = class_names

        self.construct_transform_map()

    def construct_transform_map(self):
        """
        Construct transformation maps and functions to map between class labels, indices, and names.
        """
        self.transform_map = dict(zip(self.legends, list(range(len(self.legends)))))
        self.inv_transform_map = {v: k for k, v in self.transform_map.items()}

        self.transform_func = lambda x: self.transform_map[x]
        self.inv_transform_func = lambda x: self.inv_transform_map[x]

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load a pre-trained model.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Abstract method to save the current model.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Abstract method to validate the model.
        """
        pass

    @abstractmethod
    def infer(self):
        """
        Abstract method to perform inference with the model.
        """
        pass