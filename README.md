# vitpose-classification

## Introduction

This project aims to classify VITpose images into different classes using machine learning techniques. VITpose images represent human poses captured through computer vision, and the goal is to create a model that can accurately classify these poses into predefined categories.

![Example Pose Classification]

## Table of Contents
- [Dependencies](#Depemdencies)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model Saving](#Model Saving)
- [Model Details](#model-details)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Summary](#Summary)
  
## Dependencies

To run this project, you will need the following dependencies:

- Python 3.x
- TensorFlow (version x.x)
- Numpy (version x.x)
- Scikit-learn (version x.x)

## Getting Started

1. Clone the project repository:

   ```bash
   git clone https://github.com/Harshit0933/vitpose-classification.git

2. Install the required dependencies:
   pip install tensorflow numpy scikit-learn

## Data Preparation

1. Organize your pose data in the following directory structure:

data/
├── class_1/
│   ├── file1.npy
│   ├── file2.npy
│   └── ...
├── class_2/
│   ├── file1.npy
│   ├── file2.npy
│   └── ...
└── ...

2. Update the `data_dir` variable in the script to point to your data directory.

## Usage

1. Run the script:
   ```bash
   python pose_classification.py

2. The script will load and preprocess the data, train the CNN model, and evaluate its performance.
3. Then you can run the script to prectict the unseen pose:
    ```bash
   prediction.py

## Model Saving

- The trained model will be saved as "pose_classification_model.h5" in the project directory.

## Model Details

- The CNN architecture can be customized by modifying the model definition in the script.
- You can adjust hyperparameters, such as the number of epochs and batch size, to optimize the model for your dataset.

## Results

Data shape: (40553, 1, 17, 64, 48)
Labels shape: (40553,)

Test Loss: 3.737150109373033e-05
Test Accuracy: 1.0

## Acknowledgments

- Special thanks to TensorFlow and scikit-learn communities for their powerful libraries.

## Summary
1. Method choice :
The choice of Convolutional Neural Networks (CNNs) for this pose classification task was driven by their exceptional ability to handle spatial data, making them  well-suited for processing images and heatmaps. Given that the input consists of 17 key points of the body represented in a heatmap format, preserving the spatial relationships between these points is crucial for accurate pose classification. CNNs are inherently designed to capture spatial patterns, making them a natural choice.

2. Limitations :
One of the primary limitations of this approach is the quality and diversity of the dataset. CNNs require substantial and diverse data to generalize effectively. A small or unrepresentative dataset can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.Another limitation is the potential for overfitting, which can occur when the model becomes too specialized on the training data. Regularization techniques such as dropout and data augmentation were not implemented in the code, leaving the model vulnerable to overfitting, especially with limited data.

3. How to improve :
To overcome these limitations, several strategies can be implemented. Data augmentation can be used to artificially increase the dataset's size and diversity, helping the model generalize better. Transfer learning, starting with pre-trained CNN models, can leverage knowledge from existing models and adapt it to the pose classification task. Systematic hyperparameter tuning and regularization techniques, such as dropout layers, can enhance the model's generalization capabilities and mitigate overfitting.


