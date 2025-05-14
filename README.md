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
The model used in this project is trained on 'Captcha Image Dataset' found in Kaggle.The dataset contains 10,001 images, with each file labelled after the captcha string. The data is split as 'train' and 'test' in the source. However, we had all the images commbined and stored in the same folder from which it was then split into training and testing.

- **Source** : [CAPTCHA Image Dataset](https://www.kaggle.com/datasets/johnbergmann/captcha-image-dataset/data)
- **Task** : Captcha Identification (Identify the captcha and output the string displayed in it)
- **Data** : 10,0001 captcha images wich are 6 characters long and have dimensions 250x50.
- **Labels** : The image fiiles are named after the captcha string.

## Training/Inference code and Performance visualisation

Refer the notebook for the code and visualisation : [ResNet.ipynb](ResNet.ipynb)
    

## About the Model


## Performance Metrics
- Training : CrossEntropyLoss as loss function
- Testing  : Classification accuracy as evaluation metric
