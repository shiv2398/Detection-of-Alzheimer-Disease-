# Alzheimer Detection using Deep Learning 🧠💻

## Table of Contents
1. [Introduction](#introduction)
2. [Train the Model](#train-the-model)
   - [Command](#train-command)
   - [Arguments](#train-arguments)
   - [Data Paths](#data-paths)
   - [Validation](#validation)
   - [Imbalanced Data](#imbalanced-data)
   - [Example](#train-example)
3. [Test the Model](#test-the-model)
   - [Command](#test-command)
   - [Arguments](#test-arguments)
   - [Example](#test-example)
4. [Model Testing](#model-testing)
   - [Command](#model-testing-command)
   - [Arguments](#model-testing-arguments)
   - [Example](#model-testing-example)

## Introduction
This project focuses on Alzheimer detection using deep learning techniques. The provided CLI (Command-Line Interface) allows you to train, test, and perform model testing with ease.

## Train the Model <a name="train-the-model"></a>

### Command <a name="train-command"></a>
```bash
python script_name.py train --train_data_path <train_data_path> --Epochs <num_epochs> --model <model_name> --test_data_path <test_data_path> [--val_exp] [--val_data_path <val_data_path>] [--biased]
```

### Arguments <a name="train-arguments"></a>
- `--train_data_path`: Path to the training data.
- `--Epochs`: Number of training epochs.
- `--model`: Specify the model architecture.
- `--test_data_path`: Path to the test data.

### Data Paths <a name="data-paths"></a>
- Ensure that the provided data paths exist.

### Validation <a name="validation"></a>
- Use `--val_exp` to indicate the use of explicit validation data.
- If `--val_exp` is provided, include `--val_data_path` with the validation data path.

### Imbalanced Data <a name="imbalanced-data"></a>
- Include `--biased` if the data is imbalanced.

### Example <a name="train-example"></a>
```bash
python script_name.py train --train_data_path data/train_data --Epochs 50 --model CNN_model --test_data_path data/test_data --val_exp --val_data_path data/val_data --biased
```

## Test the Model <a name="test-the-model"></a>

### Command <a name="test-command"></a>
```bash
python script_name.py test --model <model_name> --test_data_path <test_data_path> --model_path <model_path>
```

### Arguments <a name="test-arguments"></a>
- `--model`: Specify the model architecture.
- `--test_data_path`: Path to the test data.
- `--model_path`: Path to the saved model.

### Example <a name="test-example"></a>
```bash
python script_name.py test --model CNN_model --test_data_path data/test_data --model_path saved_models/CNN_model_epoch_50.h5
```

## Model Testing <a name="model-testing"></a>

### Command <a name="model-testing-command"></a>
```bash
python script_name.py model_testing --model <model_name> --test_data_path <test_data_path> --model_path <model_path>
```

### Arguments <a name="model-testing-arguments"></a>
- `--model`: Specify the model architecture.
- `--test_data_path`: Path to the test data.
- `--model_path`: Path to the saved model.

### Example <a name="model-testing-example"></a>
```bash
python script_name.py model_testing --model CNN_model --test_data_path data/test_data --model_path saved_models/CNN_model_epoch_50.h5
```
