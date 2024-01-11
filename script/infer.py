import os
import glob
import yaml
import rasterio
from natsort import natsorted
from classifier.deep_learning import (
    MSSDeepLearning,
    TMDeepLearning,
    OLITIRSDeepLearning
)
from visualizer import plot_land_cover, plot_landsat
from utils import export_geotiff
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil
from pathlib import Path
import torch
import argparse
import pandas as pd
import numpy as np

def load_model(checkpoint_path, configs_path, sensor):
    assert os.path.exists(checkpoint_path)
    assert os.path.exists(configs_path)
    
    with open(configs_path, "r") as f:
        configs = yaml.safe_load(f)
    params = {
        "architecture": configs["architecture"],
        "model_params": configs["model_params"],
        "device": configs["device"],
        "save_path": configs["save_path"],
    }
    if not torch.cuda.is_available():
        params["device"] = "cpu"
    
    model = globals()[f"{sensor}DeepLearning"](**params)
    model.load_model(checkpoint_path)
    return model

CONFIGS_PATH = {
    "MSS": "./configs/unet_dl_landsat1_megacity2035.yaml",
    "TM": "./configs/unet_dl_landsat7_megacity2035.yaml",
    "OLITIRS": "./configs/unet_dl_landsat8_megacity2035.yaml",
}

CHECKPOINT_PATH = {
    "MSS": "./checkpoints/benchmark/dummy_landsat1_best.pt",
    "TM": "./checkpoints/benchmark/dummy_landsat7_corrected-normalized_best.pt",
    "OLITIRS": "./checkpoints/benchmark/dummy_best.pt",
}

SENSOR_MAP = {
    "MSS": [1, 2, 3],
    "TM": [4, 5, 7],
    "OLITIRS": [8],
}

if __name__ == "__main__":
    # python infer.py --csv_file ./log/log.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="path to csv file")
    parser.add_argument("--output_dir", type=str, default="./prediction", help="path to predictions")
    args = parser.parse_args()

    print("Loading models...")
    sensors = ["MSS", "TM", "OLITIRS"]
    models = {}
    for sensor in sensors:
        models[sensor] = load_model(
            checkpoint_path=CHECKPOINT_PATH[sensor],
            configs_path=CONFIGS_PATH[sensor],
            sensor=sensor,
        )

    df = pd.read_csv(args.csv_file)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if (not np.isnan(row["pred_path"])) and os.path.exists(row["pred_path"]):
            continue

        # generate prediction
        for sensor, supported in SENSOR_MAP.items():
            if row["landsat"] in supported:
                model = models[sensor]

        city = row["city"]
        split = row["type"]
        img_path = row["img_path"]

        with torch.no_grad():
            preds = model.infer(img_path, convert_numpy=True)
            land_cover = model.denormalize_class(preds)

        # prepare metadata and save output
        filename = Path(img_path).name
        save_path = os.path.join(args.output_dir, split, city, filename)
        os.makedirs(Path(save_path).parent, exist_ok=True)

        output_meta = rasterio.open(img_path).meta
        output_meta["dtype"] = "uint8"
        output_meta["nodata"] = "0.0"
        params = {
            "img": land_cover,
            "save_path": save_path,
            "output_meta": output_meta,
            "compress": "PACKBITS",
        }
        try:
            export_geotiff(**params)
            df.loc[index, "pred_path"] = save_path
        except Exception:
            pass
    df.to_csv(args.csv_file, index=False)
    