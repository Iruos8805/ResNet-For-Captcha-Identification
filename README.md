# ResNet-For-Captcha-Identification
Includes a ResNet50 implmentation for captcha recognition.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Information](#dataset-information)
- [Training/Inference code and Performance visualisation](#training-/-inference-code-and-performance-visualisation)
- [About the Model](#about-the-model)
- [Performance Metrices](#performance-metrices)


## Installation

To run this project, you need to have Python installed. We recommend using a virtual environment to manage dependencies.

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

    

## Usage

1. **Download trained model**:
    [Click here to download resnet_captcha_model.pth](https://www.kaggle.com/models/souri008/resnet50-captcha-identification)


## Dataset Information
The model used in this project is trained on 'Captcha Image Dataset' found in Kaggle.The dataset contains 10,001 images, with each file labelled after the captcha sequence. The data is split as 'train' and 'test' in the source. However, we had all the images combined and stored in the same folder from which it was then split into training and testing.

- **Source** : [CAPTCHA Image Dataset](https://www.kaggle.com/datasets/johnbergmann/captcha-image-dataset/data)
- **Task** : Captcha Identification (Identify the captcha and output the sequence displayed in it)
- **Data** : 10,0001 captcha images wich are 6 characters long and have dimensions 250x50.
- **Labels** : The image fiiles are named after the captcha sequence.

## Training/Inference code and Performance visualisation

Refer the notebook for the code and visualisation : [ResNet.ipynb](ResNet.ipynb)
    

## About the Model
The project implements a ResNet50 deeo learning model for CAPTCHA rcognition. The model is designed to detect fixed-length (6 characters long) alphanumeric sequences from grayscale images. A custom ResNet architecture is used that uses bottleneck residual blocks to extract features. It predicts a seperate classification for each character in the sequence. The final layer is composed of multiple parallel classifiers, each workking for one character, predicting among 36 posssible classes (a-z, 0-9). The model is trained using label-smoothed cross-entropy loss and evaluated based on exact match accuracy, which measures whether the entire CAPTCHA is correctly predicted.


## Performance Metrics
- Training : CrossEntropyLoss as loss function
- Testing  : Classification accuracy as evaluation metric
