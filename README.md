# Signature Forgery Detection with MobileNetV3 and SVM

This project implements a signature forgery detection system using a combination of MobileNetV3 for feature extraction and Support Vector Machine (SVM) for classification. The system aims to distinguish between genuine and forged signatures based on extracted features from images.

## Overview

The project uses MobileNetV3, a lightweight convolutional neural network, to extract features from signature images. These features are then used to train an SVM classifier to differentiate between genuine and forged signatures. The system is evaluated using accuracy metrics on both validation and test datasets.

## Directory Structure

```sh
dataset/
|
├──train
│ ├── genuine/
│ │ ├── signature1.jpg
│ │ ├── signature2.jpg
│ │ └── ...
│ ├── forged/
│ ├── forged1.jpg
│ ├── forged2.jpg
│ └── ...
│
└── validation/
    ├── genuine/
    │ ├── signature3.jpg
    │ └── ...
    └── forged/
    ├── forged3.jpg
    └── ...
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- NumPy

You can install the necessary packages using pip:

```sh
pip install tensorflow opencv-python scikit-learn numpy
```

## Usage

1. Prepare the Dataset: Organize your images into the dataset directory according to the structure outlined above. Ensure that the training and validation images are placed in their respective directories.

2. Run the Script: Execute the main.py script to train and evaluate the model.

3. Prediction on New Images: To classify new signature images, update the new_image_path variable in the script to point to your new image file and re-run the script.

## Code Explanation

-Feature Extraction: MobileNetV3 is used as a feature extractor to process and convert images into feature vectors.
-Training and Validation: The dataset is split into training, validation, and test sets. An SVM classifier is trained using the training set and validated on the validation set.
-Evaluation: The model's performance is assessed using accuracy metrics on both the validation and test sets.

## Acknowledgements
-MobileNetV3: TensorFlow Keras Applications
-Support Vector Machine: scikit-learn Documentation
-Image Processing Library: OpenCV Documentation
