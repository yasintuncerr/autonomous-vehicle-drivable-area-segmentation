# LiDAR Segmentation with UNet

This project focuses on semantic segmentation of LiDAR point clouds and camera images using a UNet-based architecture. The primary goal is to accurately segment objects in autonomous driving scenes.

---
## Project Overview

The project implements a **UNet model**, adaptable for both combined LiDAR and camera data, or camera-only inputs. It includes custom dataset handlers, various loss functions tailored for segmentation tasks, and comprehensive evaluation metrics. The UNet model is designed for compatibility with the HuggingFace `transformers` library.

---
## Core Components

### 1. Datasets (`src/utils/dataset.py`) 📊
-   **`MultiViewImageDataset`**: Loads camera images, corresponding masked images (ground truth), and point cloud images. It expects data in `cam`, `masked`, and `pc` subdirectories within the `root_dir`.
-   **`JustCAM`**: A variation of `MultiViewImageDataset` that loads only camera and masked images, providing a zero tensor for the point cloud input. This is useful for training or evaluating camera-only models.

### 2. Model (`src/model/unet.py`) 🧠
-   **`UNetModel`**: A UNet architecture implemented for semantic segmentation. It inherits from `PreTrainedModel` from the HuggingFace `transformers` library, making it compatible with its ecosystem (e.g., easy saving/loading with `save_pretrained` and `from_pretrained`).
    -   **`UNetConfig`**: Configuration class (inheriting from `PretrainedConfig`) for the `UNetModel`, defining parameters like input/output channels, feature layers, and input dimensions.
    -   The model includes standard UNet components like `DoubleConv` blocks, downsampling (max pooling), and upsampling (transpose convolutions) with skip connections. It handles potential dimension mismatches during upsampling using bilinear interpolation.

### 3. Loss Functions (`src/utils/loss.py`) 📉
A suite of loss functions suitable for segmentation tasks, especially with imbalanced datasets:
-   **`DiceLoss`**: Measures overlap between predicted and target segmentation.
-   **`FocalLoss`**: Addresses class imbalance by down-weighting well-classified examples.
-   **`CombinedLoss`**: A weighted combination of `FocalLoss` and `DiceLoss` (recommended for this project).
-   **`IoULoss`**: Measures Intersection over Union, a common segmentation metric.
-   **`TverskyLoss`**: Generalizes Dice loss and IoU, allowing for balancing false positives and false negatives.
-   **`WeightedBCELoss`**: Binary Cross-Entropy loss with positive class weighting.

### 4. Metrics (`src/utils/metrics.py`) 📏
-   **`SegmentationMetrics`**: A class to compute various segmentation metrics:
    -   IoU (Intersection over Union)
    -   Dice Score (F1 Score)
    -   Precision
    -   Recall
    -   Pixel Accuracy

### 5. Preprocessing Utilities (`src/utils/preprocess.py`) 🛠️
-   **`create_pointcloud_image`**: Converts raw point cloud data (coordinates and coloring) into a 2D image representation.
-   **`create_morphological_polygon`**: Generates a binary mask from point cloud data using morphological operations (dilation and hole filling) to create solid polygon shapes.

---
## Training

The model training is primarily handled by the `notebooks/train.ipynb` script.

### Hyperparameters
The following hyperparameters were used for training:

| Hyperparameter      | Value/Type            |
| :------------------ | :-------------------- |
| Epochs              | 10                    |
| Batch Size          | 24                    |
| Learning Rate       | 1e-4                  |
| Weight Decay        | 1e-5                  |
| Optimizer           | AdamW                 |
| Input Size          | (6 x 398 x 224)       |
| Output Size         | (1 x 398 x 224)       |
| Features (layers)   | (64, 128, 256, 512)   |

*Note: Input size (6 channels) suggests concatenation of camera (3 channels) and LiDAR (e.g., 3 channels representing XYZ or other features projected to image plane) inputs.*

---
## Evaluation

Model evaluation is performed using the `notebooks/evaluate.ipynb` script, employing the metrics defined in `src/utils/metrics.py`.

### Results
The performance of the model under different input configurations is summarized below:

| Setting (Input Type)        | IoU      | Dice (F1) | Precision | Recall   | Pixel Accuracy |
| :-------------------------- | :------- | :-------- | :-------- | :------- | :------------- |
| Normal (Camera + LiDAR)     | 0.840016 | 0.913038  | 0.903365  | 0.922943 | 0.854321       |
| Normal (Camera Only)        | 0.767417 | 0.866802  | 0.868530  | 0.866370 | 0.850432       |
| Normal (LiDAR Only)         | 0.840007 | 0.913036  | 0.903904  | 0.922375 | 0.854321       |

*(The results for "Normal (LiDAR Only)" are very similar to "Normal (Camera + LiDAR)", which might indicate either a very strong LiDAR signal or a potential area for further investigation in how the inputs are combined or weighted.)*

---
## How to Use

1.  **Setup** ⚙️:
    * Ensure your Python environment is set up.
    * Install necessary libraries (e.g., `torch`, `torchvision`, `numpy`, `Pillow`, `transformers`, `scipy`). A `requirements.txt` file would be beneficial.
2.  **Data** 💾:
    * Download and preprocess the nuScenes/NuImages dataset.
    * Organize the data as expected by the `MultiViewImageDataset` class (i.e., `cam`, `masked`, and `pc` subdirectories within a root data folder, likely specified in `data/sets`).
    * Use `notebooks/create_dataset.ipynb` for guidance on dataset preparation.
3.  **Training** 🏋️‍♀️:
    * Configure hyperparameters in `notebooks/train.ipynb`.
    * Run the `notebooks/train.ipynb` notebook to train a new model. Model checkpoints and logs might be saved in `models/` or `notebooks/output/`.
4.  **Evaluation** 📈:
    * Use `notebooks/evaluate.ipynb` to evaluate a trained model. Ensure the model path and dataset path are correctly set.
5.  **Inference** 🚀:
    * Load a pre-trained model from the `models/` directory (e.g., `UNetModel.from_pretrained("./models/lidarseg_unet")`).
    * Preprocess your input data (camera images and/or LiDAR point cloud projections) as done during training.
    * Pass the input tensor through the model to get segmentation masks.