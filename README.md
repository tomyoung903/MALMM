This README is for using MALMM for video classification tasks.

## Data

### Config

`lavis/configs/datasets/DATASET_NAME/defaults_cls.yaml`:
specifies the root dir for the videos and the paths of the annotation jsons

DATASET_NAME can be e.g., scoring.

### builder

`lavis/datasets/builders/classification_builder.py`:
initializes dataset objects (train, val, test) based on config
