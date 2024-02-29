.. UrbanLC documentation master file, created by
   sphinx-quickstart on Sun Nov 19 20:54:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UrbanLC reference documentation
====================================

====================================
Introduction
====================================

UrbanLC is a Python library for land cover classification (LCC) from Landsat Images.

It features pretrained deep learning models, which are compatible with all Landsat sensors up-to-date: MSS, TM, and OLI-TIRS.
The library further contains some utility functions for analyzing and visualizing land cover maps and tutorials for for researchers and practitioners.

.. note::

    * This library is developed by **Worameth Chinchuthakun** and maintained by **Global Urban Climate Studies Lab** at Tokyo Institute of Technology


====================================
Installation
====================================

.. code:: bash

   pip install git+https://github.com/TokyoTechGUC/urbanlc

`GitHub link <https://github.com/TokyoTechGUC/urbanlc>`_.

====================================
Disclaimer
====================================

The performance of models depends on the quality of input data (Landsat surface reflectance), such as the degree of cloud coverage and geometric/radiometric calibration errors.
There is no guarantee that the predictions will reflect the actual past land covers.
Hence, users are strongly advised to verify their accuracy by comparing/ensembling with other available historical data to increase the plausibility of the hindcasts.
With this, the developers are not held responsible for any decisions based on the model.

====================================
Table of Content
====================================

.. toctree::
   :maxdepth: 2

   getting_started


.. toctree::
   :maxdepth: 1

   citation


.. toctree::
   :caption: API documentation

   api

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`