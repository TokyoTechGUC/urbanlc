# UrbanLC

<!-- # Diffusion Sampling with Momentum
[![][Arxiv]][Arxiv-link] [![][colab]][colab-link] [![][huggingface]][huggingface-link] -->

The official implementation of **Annual Past-Present Land Cover Classification from Landsat using Deep Learning for Urban Agglomerations** (2024).

**UrbanLC** is a Python library for land cover classification (LCC) from Landsat Images. It features pretrained deep learning models, which are compatible with all Landsat sensors up-to-date: MSS, TM, and OLI-TIRS. Documentation is availiable [here](https://sincostanx.github.io/UrbanLC/).

## Installation

```bash
pip install git+https://github.com/sincostanx/UrbanLC.git
```

Our library downloads Landsat images and other satellite data from [Google Earth Engine](https://developers.google.com/earth-engine/datasets) (GEE) using the official ```earthengine-api```. Please refer to this [installation guide](https://developers.google.com/earth-engine/guides/python_install) when setup your environment.

## Getting started

We provide tutorials for researchers and practitioners in ```tutorials/```, including downloading Landsat data from GEE and inferring their land cover maps. Weights for our pre-trained LCC model will be automatically downloaded depending on the type of sensor you use. For convenience, we also provide link for downloading directly here as well:

| Model     | Sensor | Landsat | Year |
|---------------|------------------|---------|-----------------|
| [LS8](https://drive.google.com/file/d/1smOhaM635ilQMOsFjlV5d-mLRBKJzfot/view?usp=sharing) | OLI-TIRS  | 8 - 9    | 2013 - Present   |
| [LS47](https://drive.google.com/file/d/1NL-rvvusxhbVCg4GkPbWANelpyIk_QF_/view?usp=sharing) | TM          | 4 - 7   | 1982 - Present |
| [LS15](https://drive.google.com/file/d/1T2dNN931VnN1EUn8b3lmZwaY3mVxHYWg/view?usp=sharing) | MSS          | 1 - 5   | 1972 - 2013 |

## Citation

TBU

## Acknowledgement

This work was supported by JSPS KAKENHI Grant Numbers JP21H04573 and JP21K14249. Data has been provided through ALOS of the Japan Aerospace Exploration Agency, ESA CCI Land Cover project, and the European Space Agency.

## License

The code and pretrained models are licensed under MIT License. See [`LICENSE.txt`](LICENSE.txt) for details.