This README is for using MALMM for video classification tasks.

## Config

`lavis/projects/malmm/cls_camera_motion.yaml`
specifies most configs other than [Dataset config](#dataset-config), including the prompt used to elicit classification labels from the model.



## Dataset

### Dataset Config

`lavis/configs/datasets/DATASET_NAME/defaults_cls.yaml`, e.g., `lavis/configs/datasets/camera_motion_cls/defaults_cls.yaml`:
specifies csv paths


### Builder

`lavis/datasets/builders/classification_builder.py`: e.g., `class CameraMotionCLSBuilder(BaseDatasetBuilder)`
initializes dataset objects (train, val, test) based on config


### Dataset Class
E.g., `lavis/datasets/datasets/camera_motion_cls_datasets.py`. Specifies `process_label()`.

``