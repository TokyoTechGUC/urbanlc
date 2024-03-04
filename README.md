# UrbanLC

<!-- # Diffusion Sampling with Momentum
[![][Arxiv]][Arxiv-link] [![][colab]][colab-link] [![][huggingface]][huggingface-link] -->

The official implementation of **Annual Past-Present Land Cover Classification from Landsat using Deep Learning for Urban Agglomerations** (2024).

**UrbanLC** is a Python library for land cover classification (LCC) from Landsat Images. It features pretrained deep learning models, which are compatible with all Landsat sensors up-to-date: MSS, TM, and OLI-TIRS. Documentation is availiable [installation guide](https://tokyotechguc.github.io/urbanlc/).

## Installation

```bash
pip install git+https://github.com/TokyoTechGUC/urbanlc
```

Our library downloads Landsat images and other satellite data from [Google Earth Engine](https://developers.google.com/earth-engine/datasets) (GEE) using the official ```earthengine-api```. Please refer to this [installation guide](https://developers.google.com/earth-engine/guides/python_install) when setup your environment.

## Getting Started

We provide tutorials for researchers and practitioners on **Google Colab**. 

1. [Download Landsat data from GEE and Infer their land cover maps](https://colab.research.google.com/drive/1QltyFvjCqHFOj1NYeAt3gBaJRoQ11lLH?usp=sharing)
2. [Fine-tune pretrained models](https://colab.research.google.com/drive/11zWrYzU4pRFxZR9FWYmI3Yx2KGH6EDiW?usp=sharing)

Weights for our pre-trained LCC model will be automatically downloaded depending on the type of sensor you use. These models are also available on [HuggingFace](https://huggingface.co/sincostanx/urbanlc/tree/main)

| Model     | Sensor | Landsat | Year |
|---------------|------------------|---------|-----------------|
| LS8 | OLI-TIRS  | 8 - 9    | 2013 - Present   |
| LS47 | TM          | 4 - 7   | 1982 - 2022 |
| LS15 | MSS          | 1 - 5   | 1972 - 2013 |

## Acknowledgement

This work was supported by JSPS KAKENHI Grant Numbers JP21H04573 and JP21K14249. Data has been provided through ALOS of the Japan Aerospace Exploration Agency, ESA CCI Land Cover project, and the European Space Agency.

## License

The code and pretrained models are licensed under MIT License. See [`LICENSE.txt`](LICENSE.txt) for details.

## Citation

```
@article{chinchuthakun2024annual,
    title={ANNUAL PAST-PRESENT LAND COVER CLASSIFICATION FROM LANDSAT USING DEEP LEARNING FOR URBAN AGGLOMERATIONS},
    author={Worameth CHINCHUTHAKUN and David WINDERL and Alvin C.G. VARQUEZ and Yukihiko YAMASHITA and Manabu KANDA},
    journal={Journal of JSCE},
    volume={12},
    number={2},
    pages={23-16151},
    year={2024},
    doi={10.2208/journalofjsce.23-16151}
}
```