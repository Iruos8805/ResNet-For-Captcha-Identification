# ResNet-For-Captcha-Identification
Includes a ResNet implmentation for captcha identification.

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
    [Click here to download resnet_captcha_model.pth](https://github.com/your-username/your-repo-name/raw/main/resnet_captcha_model.pth)


## Dataset Information

## Training/Inference code and Performance visualisation

Refer the notebook for the code and visualisation : [ResNet.ipynb](ResNet.ipynb)
    

## About the Model


## Performance Metrics
Training : CrossEntropyLoss as loss function
Testing  : Classification accuracy as evaluation metric
