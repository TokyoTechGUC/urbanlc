import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="urbanlc",
    py_modules=["urbanlc"],
    version="0.0.1",
    description="Deep-learning-based historical land cover classification tool from Landsat images",
    author="Worameth Chinchuthakun",
    packages=find_packages(),
    entry_points={
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)