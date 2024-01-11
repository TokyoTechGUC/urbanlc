import yaml
from typing import Dict, Any
from jsonargparse import ArgumentParser, ActionConfigFile
import classifier.deep_learning as model

# TODO: we can read sensor from the config file
def main(configs, sensor):
    assert sensor in ["MSS", "TM", "OLITIRS"]
    params = {
        "architecture": configs["architecture"],
        "model_params": configs["model_params"],
        "device": configs["device"],
        "save_path": configs["save_path"],
    }
    classifier = getattr(model, f"{sensor}DeepLearning")(**params)
    classifier.train(**configs["training"])

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--config", action=ActionConfigFile)
    # args = parser.parse_args()
    config = "configs/unet_dl_landsat7_megacity2035.yaml"
    with open(config, "r") as f:
        configs = yaml.safe_load(f)

    sensor = configs["training"]["dataloader_params"]["sensor_type"]
    main(configs, sensor)