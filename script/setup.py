import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="urbanlc",
    py_modules=["urbanlc"],
    version="0.0.1",
    description="test",
    author="Worameth Chinchuthakun",
    packages=find_packages(),
    entry_points={
        # "console_scripts": [
        #     "lora_add = lora_diffusion.cli_lora_add:main",
        #     "lora_pti = lora_diffusion.cli_lora_pti:main",
        #     "lora_distill = lora_diffusion.cli_svd:main",
        #     "lora_ppim = lora_diffusion.preprocess_files:main",
        # ],
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)