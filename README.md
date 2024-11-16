# Description

This is a handwritten math expression recognition and evaluation system, which is made for a University project

## Features

- With ** syntax we can create exponent operation
- Floating point with . operator ( Or operator which very small )
- Parenthesis

## Future plans

- Fraction syntax and more sign

## The Project Structure

- `desktop` folder, which contains a frontend for Desktop
- `imgsolver` folder, where the logic and the models belong
- `models`, where I store the models while I am training them
- `web`, a fontend for Web ( Something went frong, so I can't upload the frontend part of it )

## How to register the library to the local python library?

```bash
pip install -e .
```

## The training dataset

[link to the dataset](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)

Used symbols:

- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Operators**: +, - , /, *
- **Parenthesis**: (, )

Path to the dataset: `imgsolver/model_trainer_dataset`

## Used models

- **Version 1**: Simple CNN model
- **Version 2**: VGG based small model
- **Version 3**: RestNet based modell, with residual blocks
- **Version 4**: VGG based bigger modell
- **Version 5**: (Version 4 + Hu Moments) in paralell ( **removed** )

## Character extractation method from image

- Median Blur
- Inverting Color
- Skeletonization
- Morphological Dilation
- Contour Finding
- Extracting characters based on the contours box

## Important Files

### Training Files

- `imgsolver/model_trainer/my_cnn_models/cnn_utils.py`: the model creation, training, validation, data import code. LoC ~ 600
- `imgsolver/model_trainer/expr2seg_img.py`: Converting image to segments. LoC ~ 40
- `imgsolver/model_trainer/imgsolver.py`: Loading models, and evaluating experssion on images. LoC ~ 120
- `imgsolver/model_trainer/dataset_transform.ps1`: A powershell script to reduce amount of image in the dataset to a serten amount. LoC  ~ 20
- `imgsolver/model_trainer/model_source_generator.py`: Generating file to train all version of models. LoC ~ 30
- `imgsolver/model_trainer/generate_model_metrics.py`: With this script, we can generate excell file about the model metrics. LoC ~ 20
- `imgsolver/model_trainer/v?_model_source/*`: the training code for each version. LoC ~ 4 * 5 * 5 ~ 100

### Desktop Frontend Files

- `desktop/main.py`: A Tkinter Application where we can test the system. LoC ~ 200

### Models

- `models` folder contains the traing models, the newer models have cronologically bigger date

## Preview

![img](img_for_readme/preview.png)

## Dependencies

- sklearn
- tensorflow
- skimage
- opencv
- keras
- tkinter
- sympy
- numpy
- scikit-learn
- pandas
- joblib
- seaborn
