# GeoAI: Image Geolocalization Pipeline

This project contains the directory belonging the master's thesis "Aufbau zentraler Komponenten einer bildbasierten Geolokalisierungspipeline",
meaning "Composition of core components of an image based geolocalization pipeline." 

It explores an approach of utilizing interpretable semantic image features to estimate their geolocalization, e.g. cars, vegetation or street signs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataflow](#dataflow)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Execution](#execution)
- [Planned Updates](#planned-updates)
- [Used Sources](#used-sources)
- [Information for Use of Data ](#information-for-use-of-data)

---

## Project Overview

This project implements the geolocalization of images based on their semantic features.
For this, it analyzes its objects and their location within the image.
The object detection uses two yolo11m.pt models. One is pretrained and used for features that were not present in
the training data, the other is finetuned with the Mapillary Vistas Dataset. The results of both are merged for evaluation.

The dataset is based around Mapillary Open Data and consists of:
- around 29000 reference images
- around 2200 test images

The datasets are not part of the project repository (see: Information for Use of Data).
The exact image count of the implementation is dependant on the individual API calls.

---

## Dataflow

The dataflow of the core functionalities within the pipeline can be visualized.
The run order can be derived from the dependencies in the diagram, yet it is recommended to follow the instructions mentioned in [Execution](#execution).

![Datenfluss Gesamtpipeline Englisch.png](assets/Datenfluss%20Gesamtpipeline%20Englisch.png)

---

## Project Structure

The main folders within the repository cover different core aspects:

### assets

Within `assets/`, the illustrations used for the README documentation are stored.

### data

`data/` contains the evaluation results that were used in the thesis as .csv and the YOLO validation plots within val/.
After implementation and code execution, the following folders will be created:
- `data/raw`
- `data/models`
- `data/processed/vectors`
- `data/processed/yolo`
- `runs/`

The mapillary vistas dataset needs to be manually downloaded from https://www.mapillary.com/dataset/vistas to store it in data/raw/vistas.


### scripts

`scripts/` contains all core functionalities of the geolocalization system. They are grouped according to their context.

`scripts/aggregation/`: transforms, aggregates and normalizes the data from the image detection to construct a feature vector representing those
- `aggregation_controller.py`: builds raw vectors and calls normalization functions to calculate their weights for the reference set
- `aggregation_filter.py`: applies defined confidence threshold and then filters images with no valid detections
- `aggregation_normalization.py`: weights the different dimensions within the vector
- `aggregation_pyramid.py`: sets up the spatial pyramid representation with a 693 dimensional vector
- `detect_features.py`: feature detection with both yolo models

`scripts/evaluation/`: analysis of the system
- `localization_evaluation.py`: analysis of the localization results
- `yolo_evaluation.py`: analysis of the object detection results

`scripts/localization/`: setup for the reference database and retrieval calculations
- `localization_db.py`: builds up reference database. In this implementation, it is just via a numpy-ndarray structure
- `localization_retrieval.py`: calculating the nearest neighbour of a query image inside the reference database via cosine similarity (which is identical to the scalar product in this case)

`scripts/mapillary/`: consists of the scripts used for image fetching and download from the Mapillary Open Data API
- `mapillary_controller.py`: calls the main functions from `mapillary_fetch_metadata.py` and `mapillary_load_image.py`
- `mapillary_fetch_metadata.py`: fetches the image metadata for the reference set, writes corresponding metadata csv
- `mapillary_fetch_testset.py`: fetches the testset with defined offsets to the reference image clusters
- `mapillary_load_image.py`: downloads the images from the urls defined in the metadata csv

`scripts/utils/`: helper functions shared across the entire project
- `config_loader.py`: reads and converts the project settings from config.ini, used by almost every function
- `localization_math.py`: used for shared maths functions, so far only contains the haversine_km to calculate haversine distance
- `read_csv.py`: csv helper to read a file
- `write_csv.py`: csv helper to write a file
- `yolomodel_loader.py`: downloads pretrained yolo model/weights, if they are not present in local directory

`scripts/yolo/` consists of the functions used for the vistas-dataset preparation and the training of the object detection.
- `convert_training_labels_to_yolo_format.py`: Mapillary Vistas annotations are matched to YOLO labels
- `yolo.yaml`: automatically set up class mappings for the training of the YOLO model
- `yolo_modeltrainer.py`: trains a single YOLO model with the `yolo.yaml` class mappings and the hyperparameters defined in `config.ini`
- `yolo_plot_results.py`: plots learning curves of a series into one diagramm to allow direct comparisons
- `yolo_run_trainings.py`: runs an entire training series by looping through `yolo_modeltrainer.py` with the settings defined in `setup/training_configs`, correct trainingconfig defined in `config.ini`
- `yolo_aggregate_runs.py`: aggregating metrics of multiple runs (the defined trainingseries in `setup/config.ini`) into one .csv
- `yolo_yaml_builder.py`: builds yolo.yaml from the config.ini definitions


### setup

The `/setup` defines core functionalities. It enables both simple implementation and project changes without adjusting the content of `/scripts`.
It consists of the following files:

- `config.ini`: the main config file, enabling changes to the project settings.
- `requirements.txt`: all needed requirements/dependencies, that need to be installed
- `install_local_requirements.sh`: installation shell script for fast implementation of all requirements and the ROCm wheels for YOLO-training
- `training_configs/`: parent folder of all training runs for object detection training used by `yolo_run_trainings.py`

---

## Setup

### Requirements

- Python 3.10
- Linux (tested on Ubuntu 22.04)
- Mapillary Open Data token
- Optional:
  - AMD GPU with ROCm support
  - otherwise CPU execution

### Project Setup

For project setup and implementation, the following steps can be followed:
- clone or download project repository
- set up python 3.10 environment
- install requirements: `cd setup && bash install_local_requirements.sh`, needs to be started from setup folder
- download mapillary token and save it as MAPILLARY_TOKEN in system environment
- download data: manual download of mapillary vistas dataset, save it at `data/raw/vistas`
- start all scripts via `python -m scripts` ..., recommended order is presented at [Execution](#execution)

### Configuration

The configuration option is implemented via `setup/config.ini` for simple adjustments without changing components in the scripts folder.
There are multiple clusters within the configuration file, covering the following aspects

YOLO training configs, containing the categories:
- TRAIN_YOLO
- METRICS_YOLO
- PATHS_YOLO
- CATEGORIES_PANOPTIC
- CATEGORIES_YOLO_NAMES
- CATEGORIES_YOLO_PRETRAINED

Mapillary API configs, containing the categories:
- API_MAPILLARY
- PATHS_MAPILLARY
- TILING_MAPILLARY
- BBOX_MAPILLARY
- TESTSET_MAPILLARY

Feature aggregation configs, containing the categories:
- AGGREGATION
- PATHS_AGGREGATION

Evaluation configs, containing the categories:
- EVALUATION
- PATHS_EVALUATION

This setup enables the change of parameters, paths and input files. Short descriptions of the categories are written as comments within the file.

---

## Execution

This section presents the recommended execution order when using this repository. The already described prior setup is mandatory for this.

1. Object detection

a `python -m scripts.yolo.yolo_yaml_builder`

b `python -m scripts.yolo.convert_training_labels_to_yolo_format`

Depending on running a single testrun with the config.ini parameters or a set of runs defined via training_configs, one of the following functions is used:

c1 `python -m scripts.yolo.yolo_modeltrainer`

c2 `python -m scripts.yolo.yolo_run_trainings`

The object detection results in a best.pt YOLO-model. Its location is to be written into the config.ini at [PATHS_AGGREGATION] -> model_finetuned

d `python -m scripts.yolo.yolo_aggregate_runs`

e `python -m scripts.yolo.yolo_plot_results`

f `python -m scripts.evaluation.yolo_evaluation`

2. Setting up reference database

The MAPILLARY_TOKEN needs to be set up for any of the following functions to work:

a `python -m scripts.mapillary.mapillary_controller`

b `python -m scripts.aggregation.aggregation_controller`

c (optional) `python -m scripts.localization.localization_db`

The database setup is not mandatory at this point, since the localization steps will set it up themselves.

3. Running the localization and evaluating it

a `python -m scripts.mapillary.mapillary_fetch_testset`

b `python -m scripts.evaluation.localization_evaluation`

---

## Planned Updates

Right now, aside from the more fundamental changes mentioned within my thesis,
there is one core component that will probably be updated soon.

When starting this project, I was not aware of a Mapillary Open Data SDK, since I'm pretty sure there was none.
My original API fetch functions do not work anymore so I had to update them, all the while the API got more restrictive
considering its Bounding Box sizes. Towards the end of the project during documentation, I found https://leafmap.org/maplibre/mapillary/,
a fully working SDK using the Mapillary API, making most of my functionalities in /mapillary obsolete. A new research target would be the implementation
of a larger Database that represents the world and its semantic better, based around the Mapillary Dataset with the use of the leafmap Mapillary SDK.

---

## Used Sources

For the code of this python projects, the use of some core libraries and APIs was essential. This includes:

#### Mapillary

In `scripts/mapillary/mapillary_fetch_metadata.py`
- https://www.mapillary.com/developer/api-documentation?locale=de_DE#image
- https://www.mapillary.com/developer/api-documentation/faq?locale=de_DE

#### Ultralytics YOLO

In:

`scripts/aggregation/detect_features.py`

`scripts/yolo/yolo_aggregate_runs.py`

`scripts/utils/yolomodel_loader.py`
- https://docs.ultralytics.com/tasks/detect
- https://docs.ultralytics.com/guides/yolo-performance-metrics/#object-detection-metrics
- https://docs.ultralytics.com/reference/utils/downloads
- https://github.com/ultralytics/assets/releases/download/v8.3.0/

#### Numpy

In:

`scripts/aggregation/aggregation_filter.py`

`scripts/localization/localization_db.py`

`scripts/evaluation/yolo_evaluation.py`

- https://numpy.org/devdocs/reference/generated/numpy.where.html
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
- https://numpy.org/devdocs/reference/generated/numpy.corrcoef.html
- https://numpy.org/devdocs/reference/generated/numpy.log.html

#### SciPy

In `scripts/evaluation/localization_evaluation.py`
- https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.spearmanr.html


---

## Information for Use of Data

This project is based on Mapillary datasets:

- https://www.mapillary.com/dataset/vistas

The datasets themselves are not included in this repository.
They are subject to:

- CC BY-NC-SA 4.0
- additional Mapillary terms of use

License details:
- https://creativecommons.org/licenses/by-nc-sa/4.0/

- datasets must be downloaded manually from official sources
- raw Mapillary dataset are not redistributed in this repository
- trained models derived from these datasets may not be redistributed for commercial use
- this repository is intended for research purposes only

